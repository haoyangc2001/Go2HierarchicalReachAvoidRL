from dataclasses import dataclass
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def _calculate_reach_gae(
    gamma: float,
    lam: float,
    g_seq: torch.Tensor,
    value_seq: torch.Tensor,
    done_seq: torch.Tensor,
    h_seq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch port of calculate_gae_reach4 following the JAX reference."""
    device = value_seq.device
    dtype = value_seq.dtype

    horizon = g_seq.shape[0] - 1
    num_envs = g_seq.shape[1]

    if horizon <= 0:
        empty = torch.zeros(0, num_envs, device=device, dtype=dtype)
        return empty, empty

    lam_ratio = lam / max(1.0 - lam, 1e-6)
    gamma_tensor = value_seq.new_tensor(gamma)

    gae_coeffs = torch.zeros(horizon + 1, num_envs, device=device, dtype=dtype)
    value_table = torch.zeros_like(gae_coeffs)
    value_table[0] = value_seq[-1]
    prev_done = torch.zeros(num_envs, device=device, dtype=dtype)

    q_targets = torch.zeros(horizon, num_envs, device=device, dtype=dtype)
    index_mask = torch.arange(horizon + 1, device=device).unsqueeze(1)

    for idx in range(horizon - 1, -1, -1):
        done_row = done_seq[idx].to(dtype)
        done_row_unsqueezed = done_row.unsqueeze(0)
        prev_done_unsqueezed = prev_done.unsqueeze(0)

        rolled = torch.roll(gae_coeffs, shifts=1, dims=0)
        gae_coeffs = (
            rolled * lam * (1.0 - prev_done_unsqueezed)
            + rolled * lam_ratio * prev_done_unsqueezed
        ) * (1.0 - done_row_unsqueezed)
        gae_coeffs[0] = 1.0

        mask = (index_mask < (idx + 1)).to(dtype)
        disc_to_gh = gamma_tensor * value_table
        vhs_row = torch.maximum(
            h_seq[idx].unsqueeze(0),
            torch.minimum(g_seq[idx].unsqueeze(0), disc_to_gh),
        )
        vhs_row = vhs_row * mask

        coeff_sum = gae_coeffs.sum(dim=0, keepdim=True).clamp_min(1e-8)
        norm_coeffs = gae_coeffs / coeff_sum
        q_targets[idx] = (vhs_row * norm_coeffs).sum(dim=0)

        vhs_row = torch.roll(vhs_row, shifts=1, dims=0)
        vhs_row[0] = value_seq[idx + 1]
        value_table = vhs_row
        prev_done = done_row

    advantages = q_targets - value_seq[:-1]
    return advantages, q_targets


@dataclass
class ReachAvoidBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ReachAvoidBuffer:
    def __init__(self, num_envs: int, horizon: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...], device: torch.device):
        self.num_envs = num_envs
        self.horizon = horizon
        self.device = device

        obs_dim = obs_shape[0]
        act_dim = action_shape[0]

        self.observations = torch.zeros(horizon + 1, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(horizon, num_envs, act_dim, device=device)
        self.log_probs = torch.zeros(horizon, num_envs, device=device)
        self.values = torch.zeros(horizon, num_envs, device=device)
        self.advantages = torch.zeros(horizon, num_envs, device=device)
        self.returns = torch.zeros(horizon, num_envs, device=device)
        self.g_values = torch.zeros(horizon + 1, num_envs, device=device)
        self.h_values = torch.zeros(horizon + 1, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device, dtype=torch.bool)

        self.step = 0

    def clear(self) -> None:
        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        g_values: torch.Tensor,
        h_values: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_g: torch.Tensor,
        next_h: torch.Tensor,
    ) -> None:
        idx = self.step
        self.observations[idx] = obs
        self.actions[idx] = actions
        self.log_probs[idx] = log_probs
        self.values[idx] = values
        self.g_values[idx] = g_values
        self.h_values[idx] = h_values
        self.dones[idx] = dones.bool()
        self.observations[idx + 1] = next_obs
        self.g_values[idx + 1] = next_g
        self.h_values[idx + 1] = next_h
        self.step += 1

    def store_rollout(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        g_values: torch.Tensor,
        h_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        if observations.shape != self.observations.shape:
            raise ValueError("observations shape mismatch")
        if actions.shape != self.actions.shape:
            raise ValueError("actions shape mismatch")
        if log_probs.shape != self.log_probs.shape:
            raise ValueError("log_probs shape mismatch")
        if values.shape != self.values.shape:
            raise ValueError("values shape mismatch")
        if g_values.shape != self.g_values.shape:
            raise ValueError("g_values shape mismatch")
        if h_values.shape != self.h_values.shape:
            raise ValueError("h_values shape mismatch")
        if dones.shape != self.dones.shape:
            raise ValueError("dones shape mismatch")

        self.observations.copy_(observations)
        self.actions.copy_(actions)
        self.log_probs.copy_(log_probs)
        self.values.copy_(values)
        self.g_values.copy_(g_values)
        self.h_values.copy_(h_values)
        self.dones.copy_(dones.bool())
        self.step = self.horizon


    def compute_advantages(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        if self.step != self.horizon:
            raise RuntimeError("incomplete rollout stored in buffer")

        value_seq = torch.cat((self.values, last_values.unsqueeze(0)), dim=0)
        env_dones = self.dones
        safety_dones = self.h_values[:-1] >= 0
        done_seq = torch.logical_or(env_dones, safety_dones)
        adv, targets = _calculate_reach_gae(gamma, lam, self.g_values, value_seq, done_seq, self.h_values)
        self.advantages.copy_(adv)
        self.returns.copy_(targets)

    def _flat_view(self) -> ReachAvoidBatch:
        obs = self.observations[:-1].reshape(-1, self.observations.size(-1))
        actions = self.actions.reshape(-1, self.actions.size(-1))
        log_probs = self.log_probs.reshape(-1)
        values = self.values.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)
        return ReachAvoidBatch(obs, actions, log_probs, values, advantages, returns)

    def iter_batches(self, num_mini_batches: int, num_epochs: int) -> Iterator[ReachAvoidBatch]:
        data = self._flat_view()
        batch_size = data.observations.size(0)
        mini_batch_size = batch_size // num_mini_batches

        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                idx = indices[start:end]
                yield ReachAvoidBatch(
                    data.observations[idx],
                    data.actions[idx],
                    data.old_log_probs[idx],
                    data.old_values[idx],
                    data.advantages[idx],
                    data.returns[idx],
                )


class ReachAvoidPPO:
    def __init__(
        self,
        actor_critic,
        learning_rate: float = 3e-4,
        gamma: float = 0.999,
        lam: float = 0.95,
        num_learning_epochs: int = 4,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        self.device = torch.device(device)
        self.actor_critic = actor_critic.to(self.device)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.buffer = None
        self.last_value_stats = {}

    def init_storage(self, num_envs: int, horizon: int, obs_shape, action_shape) -> None:
        obs_shape = tuple(obs_shape) if isinstance(obs_shape, (list, tuple)) else (obs_shape,)
        action_shape = tuple(action_shape) if isinstance(action_shape, (list, tuple)) else (action_shape,)
        self.buffer = ReachAvoidBuffer(num_envs, horizon, obs_shape, action_shape, self.device)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            actions = self.actor_critic.act(obs)
            log_probs = self.actor_critic.get_actions_log_prob(actions)
            values = self.actor_critic.evaluate(obs).squeeze(-1)
        return actions, log_probs, values

    def update(self) -> Tuple[float, float]:
        assert self.buffer is not None

        advantages = self.buffer.advantages
        with torch.no_grad():
            values_flat = self.buffer.values.reshape(-1)
            returns_flat = self.buffer.returns.reshape(-1)
            diff = returns_flat - values_flat
            value_mean = values_flat.mean()
            value_std = values_flat.std(unbiased=False)
            return_mean = returns_flat.mean()
            return_std = returns_flat.std(unbiased=False)
            value_rmse = diff.pow(2).mean().sqrt()
            var_returns = returns_flat.var(unbiased=False)
            diff_var = diff.var(unbiased=False)
            if var_returns.item() > 1e-8:
                explained_variance = 1.0 - diff_var / var_returns
            else:
                explained_variance = torch.tensor(0.0, device=values_flat.device)
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)
            self.last_value_stats = {
                "value_mean": value_mean.item(),
                "value_std": value_std.item(),
                "return_mean": return_mean.item(),
                "return_std": return_std.item(),
                "value_rmse": value_rmse.item(),
                "explained_variance": explained_variance.item(),
                "adv_mean": adv_mean.item(),
                "adv_std": adv_std.item(),
            }
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages.copy_(advantages)

        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        batch_count = 0

        for batch in self.buffer.iter_batches(self.num_mini_batches, self.num_learning_epochs):
            obs_batch = batch.observations
            act_batch = batch.actions
            old_log_probs = batch.old_log_probs
            returns_batch = batch.returns
            old_values = batch.old_values
            adv_batch = batch.advantages

            self.actor_critic.update_distribution(obs_batch)
            log_probs = self.actor_critic.get_actions_log_prob(act_batch)
            entropy = self.actor_critic.entropy.mean()
            values = self.actor_critic.evaluate(obs_batch).squeeze(-1)

            gae_batch = -adv_batch

            ratio = torch.exp(log_probs - old_log_probs)
            loss_actor1 = ratio * gae_batch
            loss_actor2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * gae_batch
            policy_loss = -torch.min(loss_actor1, loss_actor2).mean()

            values_clipped = old_values + torch.clamp(values - old_values, -self.clip_param, self.clip_param)
            value_loss_unclipped = (values - returns_batch).pow(2)
            value_loss_clipped = (values_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            policy_loss_acc += policy_loss.item()
            value_loss_acc += value_loss.item()
            batch_count += 1

        mean_policy_loss = policy_loss_acc / max(batch_count, 1)
        mean_value_loss = value_loss_acc / max(batch_count, 1)

        self.buffer.clear()
        return mean_policy_loss, mean_value_loss











