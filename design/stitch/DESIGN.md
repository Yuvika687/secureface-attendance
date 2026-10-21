---
name: Biometric Lab Instrument
colors:
  surface: '#10141a'
  surface-dim: '#10141a'
  surface-bright: '#353940'
  surface-container-lowest: '#0a0e14'
  surface-container-low: '#181c22'
  surface-container: '#1c2026'
  surface-container-high: '#262a31'
  surface-container-highest: '#31353c'
  on-surface: '#dfe2eb'
  on-surface-variant: '#bbc9cd'
  inverse-surface: '#dfe2eb'
  inverse-on-surface: '#2d3137'
  outline: '#859397'
  outline-variant: '#3c494c'
  surface-tint: '#2fd9f4'
  primary: '#8aebff'
  on-primary: '#00363e'
  primary-container: '#22d3ee'
  on-primary-container: '#005763'
  inverse-primary: '#006877'
  secondary: '#d0bcff'
  on-secondary: '#3c0091'
  secondary-container: '#571bc1'
  on-secondary-container: '#c4abff'
  tertiary: '#ffd6a3'
  on-tertiary: '#462b00'
  tertiary-container: '#ffb13b'
  on-tertiary-container: '#6e4600'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#a2eeff'
  primary-fixed-dim: '#2fd9f4'
  on-primary-fixed: '#001f25'
  on-primary-fixed-variant: '#004e5a'
  secondary-fixed: '#e9ddff'
  secondary-fixed-dim: '#d0bcff'
  on-secondary-fixed: '#23005c'
  on-secondary-fixed-variant: '#5516be'
  tertiary-fixed: '#ffddb5'
  tertiary-fixed-dim: '#ffb957'
  on-tertiary-fixed: '#2a1800'
  on-tertiary-fixed-variant: '#643f00'
  background: '#10141a'
  on-background: '#dfe2eb'
  surface-variant: '#31353c'
typography:
  display:
    fontFamily: Space Grotesk
    fontSize: 3.5rem
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.03em
  headline-lg:
    fontFamily: Space Grotesk
    fontSize: 2.25rem
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.025em
  headline-lg-mobile:
    fontFamily: Space Grotesk
    fontSize: 1.75rem
    fontWeight: '600'
    lineHeight: '1.25'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Space Grotesk
    fontSize: 1.5rem
    fontWeight: '500'
    lineHeight: '1.3'
    letterSpacing: -0.02em
  headline-sm:
    fontFamily: Space Grotesk
    fontSize: 1.125rem
    fontWeight: '500'
    lineHeight: '1.4'
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 1rem
    fontWeight: '400'
    lineHeight: '1.6'
    letterSpacing: -0.011em
  body-md:
    fontFamily: Inter
    fontSize: 0.875rem
    fontWeight: '400'
    lineHeight: '1.5'
    letterSpacing: -0.006em
  body-sm:
    fontFamily: Inter
    fontSize: 0.75rem
    fontWeight: '400'
    lineHeight: '1.5'
    letterSpacing: 0em
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 0.875rem
    fontWeight: '500'
    lineHeight: '1.4'
    letterSpacing: 0.02em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 0.75rem
    fontWeight: '500'
    lineHeight: '1.4'
    letterSpacing: 0.04em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 0.6875rem
    fontWeight: '400'
    lineHeight: '1.3'
    letterSpacing: 0.06em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-lg: 1.5rem
  margin: 1.25rem
  margin-md: 2rem
  margin-lg: 3rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

This design system expresses the discipline, analytical rigor, and pristine clarity of mission-critical laboratory instrumentation. Built for deep optical analysis, anti-spoofing verification, and high-throughput enterprise security, the visual language merges the surgical craft of top-tier developer platforms with hardware-grade optical scanning cues.

### Personality & Emotional Response
- **Surgical Precision:** Every border, line height, and metric reflects sub-millimeter calibration and absolute zero-tolerance for latency or error.
- **Calm Authority:** A deep black spatial void illuminated by measured cyan optical feedback prevents cognitive fatigue while assuring operators of system integrity.
- **Instrumental Tactility:** Interfaces feel like physical, high-performance optoelectronic control panels rather than generic administrative dashboards.

### Visual Aesthetic
The system executes a **Dark Technical Precision** style combining:
- **Atmospheric Layering:** Deep neutral fields structured across discrete tonal micro-steps, anchored by faint radial illumination.
- **Optical Instrument Signatures:** Calibration crosses, scan bounds, reticles, micro dot-grids, and monospaced telemetry streams.
- **Delicate Boundary Work:** Hairline borders and anisotropic light catches that separate data modules without visual bulk.

## Colors

The palette establishes an ultra-deep observational canvas where chromatic light serves exclusively to indicate sensor telemetry, live status, and verification certainty.

### Surface Architecture
- **Canvas Base (`#07090D`):** The absolute foundational substrate representing the dormant sensor state.
- **Surface Layer 1 (`#0D1117`):** Base container canvas for sidebars, structural rails, and inactive background segments.
- **Surface Layer 2 (`#121821`):** Active card containers, telemetry pods, and operational workspaces.
- **Surface Layer 3 (`#18202B`):** Elevated overlays, hover states, dropdown sheets, and active inspection viewports.

### Borders & Delimiters
- **Border Default (`#1F2733`):** 1px structural framing for standard module division.
- **Border Hover (`#2C3746`):** Proximity reaction cue when cursor or gaze targets interactive modules.
- **Border Focus / Locked (`#22D3EE`):** Unambiguous active verification targeting.

### Luminescence & Accents
- **Primary Cyan (`#22D3EE`):** Core optical beam wavelength for primary actions, active scan framing, and nominal biometric locks. Emits `primary-glow` (`rgba(34, 211, 238, 0.25)`).
- **Secondary Violet (`#8B5CF6`):** Secondary classification token, multi-factor validation stages, and auxiliary machine-learning inference channels.
- **Background Radial Glow:** Sparse `radial-gradient(circle at 50% 0%, rgba(34, 211, 238, 0.08) 0%, transparent 70%)` anchored behind primary inspection viewports.

### Semantic Status (Strictly Operational)
- **Live / Verified (`#22C55E`):** Genuine human presence confirmed; mask compliance verified.
- **Spoof / Rejected (`#EF4444`):** Synthetic attempt, silicone mask, 2D screen presentation, or replay attack detected.
- **Telemetry Warning (`#F59E0B`):** Edge-case threshold, marginal lux level, or partially occluded landmarks.

## Typography

Typography enforces a strict cognitive division:
1. **Structural Headings (`Space Grotesk`):** Engineered geometric grotesque with high architectural poise for section headers, view names, and primary verification verdicts.
2. **Operational Text (`Inter`):** Highly legible, neutral workhorse for configuration instructions, user metadata, and body contextual descriptions.
3. **Telemetry & Coordinates (`JetBrains Mono`):** Fixed-width tabular execution (`font-variant-numeric: tabular-nums`) reserved for sub-pixel coordinates, confidence percentages (e.g., `99.94%`), ISO timestamps, and cryptographic device identifiers.

## Layout & Spacing

The spatial architecture functions as a rigid instrumentation chassis governed by an 8pt base grid with a 4pt micro-increment for compact telemetry matrices.

### Breakpoints & Grid Composition
- **Desktop (>= 1280px):** 12-column dynamic framework with `gutter-lg` (24px) and outer canvas margins of `margin-lg` (48px). The biometric optical stream occupies a dedicated 7-column inspection quadrant with adjacent telemetry logs taking 5 columns.
- **Tablet (768px - 1279px):** 8-column layout with `gutter` (16px) and `margin-md` (32px). Inspection feed and analytics stack vertically.
- **Mobile (< 768px):** 4-column compact frame with 16px gutters and `margin` (20px). Telemetry rows switch to high-density tabular scrolling cards.

### Structural Dot Grid
Primary analytical work surfaces utilize a CSS radial background pattern simulating physical hardware alignment beds:
- Base: `radial-gradient(#1F2733 1px, transparent 1px)` with a size of `16px 16px`.

## Elevation & Depth

Depth is defined strictly through **luminance layering** and **optical glow refraction**, completely avoiding diffuse corporate drop shadows.

### Surface Tiers
- **Tier 0 (Viewport Matrix):** `#07090D` canvas backdrop containing dot matrices and scanner alignment crossbars.
- **Tier 1 (Panels & Shells):** `#0D1117` with a uniform 1px perimeter of `#1F2733`.
- **Tier 2 (Telemetry Blocks):** `#121821` lifted by an inner top edge highlight (`inset 0 1px 0 0 rgba(255, 255, 255, 0.05)`).
- **Tier 3 (Floating HUD Overlays):** `#18202B` with 12px backdrop blur (`backdrop-filter: blur(12px)`) and a delicate gradient border fading from `#2C3746` to `#1F2733`.

### Optical Glow System
Interactive elements and verification gates generate optical emission rather than standard shadow:
- **Scan Lock:** `0 0 24px -4px rgba(34, 211, 238, 0.35)`
- **Intervention Alert:** `0 0 28px -4px rgba(239, 68, 68, 0.3)`
- **Valid Gate Pass:** `0 0 24px -4px rgba(34, 197, 94, 0.35)`

## Shapes

The design system enforces precise, restrained radii reflecting high-grade physical laboratory devices.

### Radius Distribution
- **8px (`rounded-sm`):** Internal controls, micro tags, status pills, and telemetry badges.
- **10px (`rounded-md`):** Buttons, segmented controls, input fields, and selector toggles.
- **14px (`rounded-lg`):** Cards, video inspection HUDs, and data containers.
- **20px (`rounded-xl`):** Overlaid diagnostic modals and drawer viewports.

### Signature Hardware Motifs
- **Corner Brackets:** Biometric targeting bounds feature 4-corner targeting reticles (12px long, 2px wide strokes in `#22D3EE` positioned on outer card boundaries).
- **Scan Lines:** Live video containers support a 1px scan line overlay using `linear-gradient(to bottom, transparent, rgba(34, 211, 238, 0.15), transparent)` translating smoothly across the Y-axis.

## Components

### Buttons
- **Primary Optical:** Solid `#22D3EE` fill with `#07090D` text in `Space Grotesk` (Weight 600). Radius 10px. Hover generates `primary-glow` and brightness boost. Active state compresses scale to `0.98`.
- **Secondary Instrument:** Surface `#121821`, 1px border `#1F2733`, text `#F1F5F9`. Hover transitions border to `#2C3746` and shifts background to `#18202B`.
- **Ghost Action:** Borderless, text `#94A3B8`. Hover introduces text `#22D3EE` and background `rgba(34, 211, 238, 0.06)`.

### Inputs & Selectors
- Background `#0D1117`, border 1px `#1F2733`, radius 10px, text in `Inter` 14px.
- Focus state creates an instantaneous 1px border `#22D3EE` accompanied by `0 0 0 1px #22D3EE` and subtle outer halo `rgba(34, 211, 238, 0.15)`.
- Labels styled in `JetBrains Mono` 11px uppercase (`letter-spacing: 0.05em`) in `#64748B`.

### Status Indicators & Chips
- Monospaced badges styled in `JetBrains Mono` 11px.
- Encapsulated in a pill format (height 22px, radius 8px) with a 6px pulsing LED indicator on the left.
- **Live Status:** Background `rgba(34, 197, 94, 0.1)`, text `#22C55E`, border `rgba(34, 197, 94, 0.25)`.
- **Spoof Alert:** Background `rgba(239, 68, 68, 0.1)`, text `#EF4444`, border `rgba(239, 68, 68, 0.25)`.

### Telemetry Cards
- Surface `#121821`, radius 14px, border 1px `#1F2733`.
- Upper right corner hosts subtle alignment cross-hairs (`+`).
- Top interior edge highlights sub-surface depth through a 1px anisotropic sheen `linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent)`.

### Biometric Scan HUD Frame
- Centered container on live camera streams with 14px border radius.
- Four cyan corner brackets dynamically adjusting dimension based on facial bounding coordinates.
- Sub-labels render live landmark tracking metadata (`PUPIL_DIST`, `LIVENESS_CONF`, `MASK_PROB`) rendered in `label-sm` along the lower rim.