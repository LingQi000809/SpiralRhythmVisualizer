import { useRef, useEffect } from "react";


function getClusterColor(clusterIndex) {
  const hue = (clusterIndex * 47) % 360; // prime-ish step avoids repetition
  const saturation = 70;
  const lightness = 65;
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

export default function SpiralRhythmVisualizer({ audioRef, rhythmData }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    if (!audioRef?.current || !rhythmData?.onsets?.length) return;

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
      const audioDuration = audio.duration;
      const secondsPerOrbit = audioDuration < 30 ? audioDuration : 30;

      const numClusters = Math.max(
        ...rhythmData.onsets.map((o) => o.cluster)
      ) + 1; // cluster indices start at 0


      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const cx = w / 2;
      const cy = h / 2;

      const baseRadius = Math.min(w, h) * 0.1;
      const maxRadius = Math.min(w, h) * 0.4;

      // background
      ctx.fillStyle = "#0b0e14";
      ctx.fillRect(0, 0, w, h);
      // subtle galactic haze
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxRadius);
      grad.addColorStop(0, "rgba(255,255,255,0.04)");
      grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      // show glowing pulses as audio plays
      rhythmData.onsets.forEach((onset) => {
        const dt = t - onset.time;
        if (dt < -0.1 || dt > 1.5) return;

        const orbitProgress =
          (onset.time % secondsPerOrbit) / secondsPerOrbit;
        const angle = orbitProgress * Math.PI * 2 - Math.PI / 2;

        const clusterOffset = (onset.cluster / numClusters); // 0 -> 1
        const centroidNorm = Math.min(onset.centroid / 8000, 1);

        const r =
          baseRadius +
          clusterOffset * (maxRadius * 1) +
          centroidNorm * (maxRadius * 0.2);

        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;

        const life = Math.max(0, 1 - dt / 1.2);
        const size = 2 + centroidNorm * 5;
        const glow = 10 + centroidNorm * 15;

        // color
        const clusterColor = getClusterColor(onset.cluster);
        ctx.beginPath();
        ctx.fillStyle = clusterColor;
        ctx.globalAlpha = life;
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();

        // glow
        ctx.beginPath();
        ctx.globalAlpha = life * 0.25;
        ctx.fillStyle = clusterColor;
        ctx.arc(x, y, glow, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.globalAlpha = 1;
      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [audioRef, rhythmData]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: "100%",
        height: "100%",
        display: "block",
        background: "#0b0e14",
      }}
    />
  );
}