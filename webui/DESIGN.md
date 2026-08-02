# HD-MCRA Academic Project Page Reference

## 1. Visual Theme & Atmosphere

An academic project page inspired by the information hierarchy of the Academic Project Astro Template: calm, paper-like, media-led, and easy to scan. It presents HD-MCRA as a research contribution rather than a product landing page.

## 2. Color Palette & Roles

```css
:root {
  --canvas: #fbfcfc;
  --paper: #ffffff;
  --ink: #1f2933;
  --muted: #607080;
  --line: #dbe3e7;
  --teal: #007c78;
  --teal-soft: #e8f5f3;
  --orange: #dc5f21;
  --code: #13242b;
}
```

## 3. Typography Rules

- Chinese body: Noto Sans SC, Microsoft YaHei, sans-serif.
- English display text: Inter, Noto Sans SC, sans-serif.
- Paper title: 42px desktop, 31px mobile; normal weight and zero negative letter spacing.
- Section title: 29px desktop, 24px mobile.
- Body: 16px with a 1.8 line height.

## 4. Component Styling

- Header: centered title, author line, institution line, quiet outline links.
- Highlighted abstract: a full-width soft-teal band with a narrow readable inner column.
- Video and figures: square 6px corners, pale borders, no floating-card shadow. Result figures preserve their natural image ratio and remain smaller than the video media frame.
- Result chips: restrained inline badges; numerical values receive the accent color.
- Method nodes: a centered, narrow top-to-bottom sequence with separators at five-sixths of the content width. Each node places its number and title on the first line and left-aligned explanation beneath the title, followed by the annotated feasibility-driven policy figure.
- Footer: minimal project identity with navigation to the page top and the public source repository.

## 5. Layout Principles

- Main text column: 760px.
- Wide media column: 1080px.
- Outer page padding: 28px desktop and 18px mobile.
- Sections follow title, short explanatory text, then supporting media or evidence.

## 6. Depth & Elevation

Use borders and tone changes instead of shadows. Only video controls and the copy button may use a slight elevation on hover.

## 7. Animation & Interaction

- L2 interaction level: sections reveal with a 20px upward fade on intersection.
- Navigation links scroll smoothly to sections.
- BibTeX copy button confirms completion in place.
- `prefers-reduced-motion` disables motion and smooth scrolling.

## 8. Do's and Don'ts

- Do prioritize real video and experimental evidence.
- Do keep the writing column narrow for long technical passages.
- Do make formulas and labels readable as direct inline-page content, without decorative containers or horizontal scrolling.
- Do use a single accent pair: teal for structure, orange for critical results.
- Do preserve Chinese-first technical narration.
- Don't use a full-screen video background.
- Don't turn methods into oversized marketing cards.
- Don't use gradients, bokeh, or decorative 3D objects.
- Don't use oversized numeric counters.
- Don't add unsupported claims, paper links, or author names.

## 9. Responsive Behavior

- At 760px, methods and evidence layouts stack vertically.
- At 560px, title, author metadata, and external-link buttons wrap naturally.
- All videos retain aspect ratio and fit the viewport width.
- Interactive targets remain at least 44px high on touch devices.
