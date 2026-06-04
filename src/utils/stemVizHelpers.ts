// Stem visual constants and ring-layout helpers for StemVisualizationView.
// Hex colors are identical to Hearmi-Frontend/app/studio/components/vizHelpers.ts STEM_HEX.

/** Hex colors per stem — identical to production vizHelpers.ts */
export const STEM_HEX: Record<string, string> = {
  drums:   '#FF3366',
  bass:    '#FF6633',
  vocals:  '#33CCFF',
  other:   '#9933FF',
  silence: '#333333',
};

/**
 * Default stem order, inner ring (index 0) to outer ring (index 3).
 * Matches the user preference: other → bass → drums → vocals (outermost).
 */
export const DEFAULT_STEM_ORDER: string[] = ['bass', 'drums', 'other', 'vocals'];

// Radial extents for the ring layout, relative to baseRadius.
// Chosen so the inner ring is clearly visible and the outer ring has room for labels.
const INNER_FACTOR = 0.55;
const OUTER_FACTOR = 2.3;

/**
 * Center orbital radius for the ring at `ringIdx` (0 = innermost).
 * Rings are evenly spaced between INNER_FACTOR and OUTER_FACTOR times baseR.
 */
export function ringCenterRadius(ringIdx: number, numRings: number, baseR: number): number {
  if (numRings <= 1) return baseR;
  const t = ringIdx / (numRings - 1);
  return baseR * (INNER_FACTOR + t * (OUTER_FACTOR - INNER_FACTOR));
}

/**
 * Half-width of the pitch-spread band around a ring center.
 * Notes at the stem's median pitch land exactly on the ring line;
 * notes ±1 octave away deviate by at most this amount.
 * Set to 30% of the inter-ring gap so adjacent bands barely overlap.
 */
export function ringBandHalf(numRings: number, baseR: number): number {
  if (numRings <= 1) return baseR * 0.4;
  const spacing = baseR * (OUTER_FACTOR - INNER_FACTOR) / (numRings - 1);
  return spacing * 0.3;
}

/**
 * Draws a faint orbit circle for a stem ring.
 * Selected rings are slightly brighter; drag-target rings show as dashed.
 */
export function drawOrbitRing(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, r: number,
  hexColor: string,
  isSelected: boolean,
  isDragTarget: boolean
): void {
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = hexColor;
  ctx.globalAlpha = isDragTarget ? 0.45 : isSelected ? 0.15 : 0.08;
  ctx.lineWidth   = isDragTarget ? 1.5 : 0.5;
  if (isDragTarget) ctx.setLineDash([6, 5]);
  ctx.stroke();
  ctx.restore();
}

/**
 * Draws the stem name label just outside the ring at the rightmost point (angle = 0).
 */
export function drawStemLabel(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, r: number,
  stem: string, hexColor: string,
  isSelected: boolean, isDimmed: boolean
): void {
  ctx.save();
  ctx.globalAlpha = isDimmed ? 0.18 : isSelected ? 0.95 : 0.55;
  ctx.fillStyle = hexColor;
  ctx.font = `${isSelected ? 'bold ' : ''}11px Inter, sans-serif`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(stem.toUpperCase(), cx + r + 10, cy);
  ctx.restore();
}

/**
 * Converts a stem hex color into an hsla string modulated by energy/brightness.
 * Keeps the stem's hue identity while letting rms/centroid drive glow intensity.
 */
// Returns a stem-tinted galaxy color driven by energy/brightness.
// dimming is handled by the caller via alphaScale — do NOT bake dimAlpha in here.
export function stemGalaxyColor(stem: string, rms: number, centroid: number): string {
  const stemHsl: Record<string, [number, number]> = {
    drums:   [340, 85],
    bass:    [22,  90],
    vocals:  [195, 85],
    other:   [280, 85],
    silence: [0,   0],
  };
  const [h, s] = stemHsl[stem] ?? [0, 0];
  if (s === 0) return `rgba(80,80,80,0.1)`;
  const lightness = 42 + centroid * 22; // 42–64%
  const alpha     = 0.3 + rms * 0.5;   // same range as getGalaxyColor
  return `hsla(${h}, ${s}%, ${lightness}%, ${alpha})`;
}
