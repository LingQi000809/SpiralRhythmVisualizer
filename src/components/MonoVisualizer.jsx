import { useRef, useEffect } from "react";

// Shape helpers
function drawTriangle(ctx, x, y, size) {
  ctx.beginPath();
  ctx.moveTo(x, y - size);
  ctx.lineTo(x - size, y + size);
  ctx.lineTo(x + size, y + size);
  ctx.closePath();
  ctx.fill();
}

function drawStar(ctx, x, y, size) {
  const spikes = 5;
  const outer = size;
  const inner = size / 2;
  let rot = Math.PI / 2 * 3;
  const cx = x, cy = y;
  const step = Math.PI / spikes;

  ctx.beginPath();
  ctx.moveTo(cx, cy - outer);
  for (let i = 0; i < spikes; i++) {
    ctx.lineTo(cx + Math.cos(rot) * outer, cy + Math.sin(rot) * outer);
    rot += step;
    ctx.lineTo(cx + Math.cos(rot) * inner, cy + Math.sin(rot) * inner);
    rot += step;
  }
  ctx.closePath();
  ctx.fill();
}

function getGalaxyColor(rms, centroid) {
  // console.log("rms, centroid", rms, centroid) //0.5, 0.8; 0.02, 2575
  const hue = 220 + centroid * 80; // bluish-purple
  const sat = 40 + centroid * 40;
  const light = 50 + centroid * 20;
  const alpha = 0.3 + rms * 0.5;
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
}

function drawNote(ctx, x, y, size, glowSize, color, alpha) {
  // glow
  ctx.globalAlpha = alpha * 0.5;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, glowSize, 0, Math.PI*2);
  ctx.fill();

  // core
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, size, 0, Math.PI*2);
  ctx.fill();
}

export default function MonoVisualizer({ audioRef, data }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    console.table(data.events)
  }, [data]);
  
  useEffect(() => {
    if (!audioRef?.current || !data?.events?.length) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.clientWidth * dpr;
      canvas.height = canvas.clientHeight * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    function draw() {
      const audio = audioRef.current;
      if (!audio.duration) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const t = audio.currentTime;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const baseRadius = Math.min(w, h) * 0.3;

      // --- Galaxy background ---
      ctx.fillStyle = "#0b0e14";
      ctx.fillRect(0, 0, w, h);

      // --- Draw musical points ---
      data.events.forEach((evt, idx) => {
        const dt = t - evt.time;
        const minAlpha = 0.02
        let life;
        if (dt < 0) life = 0;
        else if (dt < 1.2) life = 1 - dt / 1.2;
        else life = minAlpha;
        if (life === 0) return;
        
        // Within any given audio, the audio feature is normalized 0-1.
        const rms = evt.rms;
        const centroid = evt.centroid;

        // Map pitch to distance from center (normalized 0–1)
        const pitchNorm = evt.pitch / 127; // MIDI 0-127

        const rBase = baseRadius + (pitchNorm - 0.5) * baseRadius;
        const angle = ((idx / data.events.length) * Math.PI * 2) + t * 0.05; // slowly rotate
        const size = 6 + rms * 10;
        const glowSize = size * 2 + rms * 10;
        const color = getGalaxyColor(rms, centroid);
        
        // console.log(`Event ${idx}: pitch=${evt.pitch}, duration=${evt.duration}, rms=${rms}, rBase=${rBase}, angle=${angle}`);
        
        const trailSteps = Math.max(Math.floor(evt.duration * 200), 1);
        for (let j = 0; j < trailSteps; j++) {
            const trailAlpha = Math.max(life * (1 - j / trailSteps), minAlpha);
            const trailSize = size * (1 - j / trailSteps);
            const trailGlow = glowSize * (1 - j / trailSteps);
            const trailR = rBase + (rms - 0.5) * 20;
            const trailAngle = angle - j * 0.005;

            const x = cx + Math.cos(trailAngle) * trailR;
            const y = cy + Math.sin(trailAngle) * trailR;

            drawNote(ctx, x, y, trailSize, trailGlow, color, trailAlpha);
        }
      });

      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [audioRef, data]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: "100%", display: "block" }}
    />
  );
}
