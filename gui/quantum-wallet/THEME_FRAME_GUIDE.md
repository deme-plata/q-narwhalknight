# Theme Frame Preparation Guide

How to convert raw frame images into transparent-center PNGs ready for the Q-NarwhalKnight wallet theme system.

## Overview

The wallet uses a 3-layer rendering system:
- **Layer 1** (z-index 1): Full frame image BEHIND the UI content (atmosphere/background)
- **Layer 2** (z-index 5): Interactive UI content
- **Layer 3** (z-index 9990): Same frame image ON TOP, but CSS-masked to only reveal corners + center emblems

For this to work, each frame PNG needs a **transparent center** so the UI content shows through Layer 1.

## Prerequisites

- ImageMagick installed (`apt install imagemagick` or `brew install imagemagick`)
- A source frame image (screenshot or design, any resolution)
- The alpha mask template (extracted from the original purple frame)

## Directory Structure

```
gui/quantum-wallet/public/borders/
  purple/
    frame-full.png      # 1536x1024, TrueColorAlpha (transparent center)
  inferno/
    frame-full.png
  predator/
    frame-full.png
  ... (one directory per theme)
```

Each theme directory must contain a `frame-full.png` at **1536x1024** with an **alpha channel** (transparent center).

## Step-by-Step Process

### Step 1: Get the Alpha Mask Template

The purple frame has the reference alpha channel. Extract it once:

```bash
cd gui/quantum-wallet/public/borders

# Extract alpha channel from the original purple frame
convert purple/frame-full.png -alpha extract /tmp/purple_alpha_mask.png
```

This creates a grayscale mask: white = opaque (frame border), black = transparent (center where UI shows).

### Step 2: Prepare Your Source Image

Your source image (e.g., `frame12a.png` in `/opt/orobit/shared/q-narwhalknight/gui/`) is typically a full screenshot at whatever resolution. It needs to be:
1. Resized to 1536x1024
2. Have the alpha mask applied

```bash
# Resize source to exact frame dimensions (stretch to fill)
convert /path/to/your-frame.png -resize 1536x1024! /tmp/frame_resized.png
```

The `!` flag forces exact dimensions (ignoring aspect ratio). This is needed because the frame must fill the entire viewport.

### Step 3: Apply the Alpha Mask

Composite the resized frame with the alpha mask to create a transparent center:

```bash
# Apply alpha mask: frame pixels where mask is white, transparent where mask is black
convert /tmp/frame_resized.png /tmp/purple_alpha_mask.png \
  -alpha Off -compose CopyOpacity -composite \
  borders/mytheme/frame-full.png
```

### Step 4: Verify the Result

Check that the output has an alpha channel:

```bash
identify -verbose borders/mytheme/frame-full.png | grep -E "Type:|Geometry:"
# Should show: Type: TrueColorAlpha
# Should show: Geometry: 1536x1024
```

Or view it visually:
```bash
# Open in image viewer - center should be transparent (checkered)
display borders/mytheme/frame-full.png
```

### Complete One-Liner

For convenience, here's the entire process as a single command:

```bash
# Replace THEME_NAME and SOURCE_IMAGE with your values
THEME_NAME="mytheme"
SOURCE_IMAGE="/path/to/frame.png"

mkdir -p gui/quantum-wallet/public/borders/${THEME_NAME} && \
convert ${SOURCE_IMAGE} -resize 1536x1024! /tmp/frame_resized.png && \
convert gui/quantum-wallet/public/borders/purple/frame-full.png -alpha extract /tmp/alpha_mask.png && \
convert /tmp/frame_resized.png /tmp/alpha_mask.png -alpha Off -compose CopyOpacity -composite \
  gui/quantum-wallet/public/borders/${THEME_NAME}/frame-full.png && \
echo "Done: borders/${THEME_NAME}/frame-full.png created"
```

## Adding the Theme to the Frontend

### 1. Update `AnimatedBorder.tsx`

Add the new theme ID to the `BorderTheme` type:

```tsx
export type BorderTheme =
  | 'purple'
  | 'predator'
  | 'mytheme'    // <-- add here
  | 'red'
  // ...
```

Add a theme config entry to `THEME_LIST`:

```tsx
{
  id: 'mytheme', name: 'My Theme', accent: '#ff6600', accentAlt: '#ff9933',
  textPrimary: '#fff0e0', textSecondary: '#ffb366', borderGlow: '#cc5200',
  bgCard: 'rgba(100, 40, 0, 0.15)', font: '"Orbitron", sans-serif',
},
```

**Theme config fields:**
| Field | Purpose | Example |
|-------|---------|---------|
| `id` | Directory name under `borders/`, localStorage key | `'mytheme'` |
| `name` | Display name in theme chooser | `'My Theme'` |
| `accent` | Primary accent color (buttons, highlights) | `'#ff6600'` |
| `accentAlt` | Secondary accent (gradients, alt highlights) | `'#ff9933'` |
| `textPrimary` | Heading/balance text color | `'#fff0e0'` |
| `textSecondary` | Labels/muted text color | `'#ffb366'` |
| `borderGlow` | Border/separator glow color | `'#cc5200'` |
| `bgCard` | Card/panel background tint (use rgba with low alpha) | `'rgba(100, 40, 0, 0.15)'` |
| `font` | Google Font family (must be loaded) | `'"Orbitron", sans-serif'` |

### 2. Add Google Font (if using a new one)

In `AnimatedBorder.tsx`, add the font to the `fonts` array in the `useEffect` that loads Google Fonts:

```tsx
const fonts = [
  'Cinzel:wght@400;700',
  'MyNewFont:wght@400;600;700',  // <-- add here
  // ...
];
```

### 3. Bump FRAME_VERSION

Increment `FRAME_VERSION` in `AnimatedBorder.tsx` to bust browser cache:

```tsx
export const FRAME_VERSION = 4;  // was 3
```

### 4. (Optional) Custom Corner Masks

If your frame has unique corner decorations that need special sizing, add a CSS override in `AnimatedBorder.css`:

```css
/* Example: larger top corners for frames with big corner art */
[data-theme="mytheme"] .border-frame-corners {
  -webkit-mask-image:
    radial-gradient(ellipse 42% 42% at 0% 0%, black 50%, transparent 100%),
    radial-gradient(ellipse 42% 42% at 100% 0%, black 50%, transparent 100%),
    radial-gradient(ellipse 30% 40% at 0% 100%, black 50%, transparent 100%),
    radial-gradient(ellipse 30% 40% at 100% 100%, black 50%, transparent 100%),
    radial-gradient(ellipse 20% 18% at 50% 0%, black 50%, transparent 100%),
    radial-gradient(ellipse 20% 22% at 50% 100%, black 50%, transparent 100%);
  -webkit-mask-composite: source-over;
  mask-image: /* same as above */;
  mask-composite: add;
}
```

**Mask parameters explained:**
- `ellipse W% H% at X% Y%` - size and position of revealed area
- Increase W%/H% to show more of that corner
- `50% 0%` = top center, `50% 100%` = bottom center
- Add `linear-gradient(to bottom, black 0%, black 5%, transparent 8%)` to reveal the full top edge bar

### 5. (Optional) Custom Content Padding

If your frame has thick borders, narrow the content area:

```css
[data-theme="mytheme"] .animated-border-container > .border-content {
  padding: 7% 11% 18% 11%;  /* default is 6% 8% 17% 8% */
}
```

### 6. (Optional) Theme-Specific CSS Effects

Add special visual effects in `index.css` for your theme:

```css
[data-theme="mytheme"] body {
  background: #0a0500 !important;
}

[data-theme="mytheme"] .text-amber-400 {
  text-shadow: 0 0 8px rgba(255, 102, 0, 0.4);
}
```

## Build and Deploy

```bash
# 1. Build
cd gui/quantum-wallet
npx vite build

# 2. Copy frame to dist-final (vite copies public/ automatically, but verify)
cp -r public/borders/mytheme dist-final/borders/mytheme

# 3. Reload nginx
nginx -s reload

# 4. Hard refresh browser (Ctrl+Shift+R) to bypass cache
```

## Existing Theme → Frame Mapping

| Theme ID | Source Frame | Description |
|----------|------------|-------------|
| `purple` | Original | Ornate gold with purple gems |
| `predator` | `frame3a.png` | Dark biomechanical, green accents |
| `inferno` | `frame1a.png` | Alien heads, red/orange glow |
| `crimson` | `frame2a.png` | Red steel mechanical |
| `emerald` | `frame4a.png` | Green crystalline |
| `abyssal` | `frame5a.png` | Deep teal oceanic |
| `neon` | `frame6a.png` | Neon pink/cyan cyber |
| `aurora` | `frame7a.png` | Blue aurora borealis |
| `hellfire` | `frame8a.png` | Orange hellfire flames |
| `alien` | `frame9a.png` | Green alien technology |
| `sovereign` | `frame10a.png` | Gold sovereign ornate |
| `frost` | `frame11a.png` | Ice crystal blue |
| `cosmic` | `frame1a.png` (variant) | Purple cosmic void |
| `red` | Original | Blood red (transaction flash only, not in chooser) |

## Troubleshooting

**Frame covers the UI content (no transparency):**
- Source image has no alpha channel. Re-apply the alpha mask (Step 3).
- Verify with `identify -verbose file.png | grep Type` - must be `TrueColorAlpha`.

**Old frame still showing after update:**
- Bump `FRAME_VERSION` in `AnimatedBorder.tsx` and rebuild.
- Or hard refresh with Ctrl+Shift+R.
- Nginx caches PNGs for 1 hour by default.

**Corner details hidden behind UI:**
- The CSS mask in Layer 3 controls what shows on top. Increase the radial gradient size for that corner.
- Default: `ellipse 30% 30%` - try `ellipse 40% 40%` for bigger reveals.

**Content too wide/narrow for frame:**
- Add a `[data-theme="mytheme"] .animated-border-container > .border-content` padding override.
- Increase side padding (3rd and 4th values) to narrow content.
