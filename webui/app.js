const nav = document.querySelector('[data-nav]');
const navToggle = document.querySelector('[data-nav-toggle]');
const navMenu = document.querySelector('[data-nav-menu]');
const progress = document.querySelector('.scroll-progress');
const policyFlow = document.querySelector('[data-policy-flow]');
const modeControls = document.querySelectorAll('[data-mode]');

function updateScrollChrome() {
  nav?.classList.toggle('is-scrolled', window.scrollY > 48);
  const height = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.transform = `scaleX(${height > 0 ? window.scrollY / height : 0})`;
}

function revealOnView() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.14 });

  document.querySelectorAll('.reveal').forEach((element) => observer.observe(element));
}

navToggle?.addEventListener('click', () => {
  const isOpen = navMenu.classList.toggle('is-open');
  navToggle.setAttribute('aria-expanded', String(isOpen));
  navToggle.querySelector('i').setAttribute('data-lucide', isOpen ? 'x' : 'menu');
  lucide.createIcons();
});

navMenu?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    navMenu.classList.remove('is-open');
    navToggle?.setAttribute('aria-expanded', 'false');
  });
});

modeControls.forEach((control) => {
  control.addEventListener('click', () => {
    const mode = control.dataset.mode;
    policyFlow.dataset.mode = mode;
    modeControls.forEach((item) => {
      const active = item === control;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-selected', String(active));
    });
  });
});

document.addEventListener('DOMContentLoaded', () => {
  lucide.createIcons();
  revealOnView();
  updateScrollChrome();
  window.addEventListener('scroll', updateScrollChrome, { passive: true });
});
