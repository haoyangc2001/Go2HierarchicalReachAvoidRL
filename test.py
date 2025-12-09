import pandas as pd
import torch

snap = torch.load('logs/high_level_go2/20251003-063321/gh_snapshots/gh_iter_00050.pt')
g = snap["g_values"].cpu()
h = snap["h_values"].cpu()
dones = snap["dones"].cpu()

def dump_tensor(tensor, filename):
    df = pd.DataFrame(tensor.numpy())
    df.index.name = "step"
    df.columns = [f"env_{i}" for i in range(tensor.shape[1])]
    df.to_csv(filename, sep="\t", float_format="%.6f")

dump_tensor(g, "g_values.tsv")
dump_tensor(h, "h_values.tsv")
dump_tensor(dones, "dones.tsv")

print("导出完成: g_values.tsv / h_values.tsv / dones.tsv")
