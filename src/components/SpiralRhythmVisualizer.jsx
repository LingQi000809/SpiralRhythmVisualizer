import { useRef, useEffect } from "react";


function getClusterColor(clusterIndex) {
  const hue = (clusterIndex * 47) % 360; // prime-ish step avoids repetition
  const saturation = 70;
  const lightness = 65;
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

function summarizeClusters(onsets) {
  const clusters = {};

  onsets.forEach(o => {
    if (!clusters[o.cluster]) {
      clusters[o.cluster] = {
        centroidSum: 0,
        count: 0,
      };
    }
    clusters[o.cluster].centroidSum += o.centroid;
    clusters[o.cluster].count += 1;
  });

  return Object.entries(clusters).map(([cluster, v]) => ({
    cluster: Number(cluster),
    meanCentroid: v.centroidSum / v.count,
  }));
}

function describeClusters(clusterStats) {
  // rank by brightness (centroid)
  const sorted = [...clusterStats].sort(
    (a, b) => a.meanCentroid - b.meanCentroid
  );

  return sorted.map((c, i) => {
    const p = i / (sorted.length - 1 || 1);

    let brightness;
    if (p < 0.2) brightness = "dark";
    else if (p < 0.4) brightness = "warm";
    else if (p < 0.6) brightness = "neutral";
    else if (p < 0.8) brightness = "bright";
    else brightness = "sharp";

    return {
      cluster: c.cluster,
      description: brightness,
    };
  });
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

    const clusterStats = summarizeClusters(rhythmData.onsets);
    const clusterDescriptions = describeClusters(clusterStats).sort(
      (a, b) => a.cluster - b.cluster
    );


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


      const w = canvas.clientWidth;
      const h = canvas.clientHeight;

      ctx.clearRect(0, 0, w, h);

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
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      // show glowing pulses as audio plays
      rhythmData.onsets.forEach((onset) => {
        const dt = t - onset.time;

        const orbitProgress =
          (onset.time % secondsPerOrbit) / secondsPerOrbit;
        const orbitCount = Math.floor(onset.time / secondsPerOrbit);
        const angle = orbitProgress * Math.PI * 2 - Math.PI / 2;

        const clusterOffset = (onset.cluster / numClusters); // 0 -> 1
        const centroidNorm = Math.min(onset.centroid / 8000, 1);

        const r =
          baseRadius +
          clusterOffset * (maxRadius * 1) +
          centroidNorm * (maxRadius * 0.2);

        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;

        let life;
        if (dt < 0) {
          life = 0;
        } else if (dt < 1.2) {
          life = 1 - dt / 1.2; // current pulse
        } else {
          // temporal trace floor decays per orbit
          life = 0.1 * Math.pow(0.5, orbitCount); // halves the alpha every full rotation
        }


        const size = 2 + centroidNorm * 5;
        const glow = 10 + centroidNorm * 15;
        const clusterColor = getClusterColor(onset.cluster);

        // color
        if (dt >= 0 && dt < 1.2) {
          ctx.globalAlpha = life;
          ctx.fillStyle = clusterColor;
          ctx.beginPath();
          ctx.arc(x, y, size, 0, Math.PI * 2);
          ctx.fill();
        }     

        // glow
        ctx.beginPath();
        ctx.globalAlpha = life * 0.25;
        ctx.fillStyle = clusterColor;
        ctx.arc(x, y, glow, 0, Math.PI * 2);
        ctx.fill();
      });

      // ---------- LEGEND ----------
      const padding = 12;
      const lineHeight = 22;
      const startX = padding;

      const headerHeight = 22;
      const startY = padding + headerHeight;

      // explanatory line
      ctx.globalAlpha = 0.9;
      ctx.font = "14px system-ui, sans-serif";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.fillText(
        "Colors = similar sound textures",
        startX,
        padding + 6
      );

      // legend items
      ctx.globalAlpha = 0.9;
      ctx.font = "14px system-ui, sans-serif";

      clusterDescriptions.forEach((d, i) => {
        const y = startY + i * lineHeight;

        // color dot
        ctx.fillStyle = getClusterColor(d.cluster);
        ctx.beginPath();
        ctx.arc(startX + 4, y + 6, 4, 0, Math.PI * 2);
        ctx.fill();

        // text
        ctx.fillStyle = "rgba(255,255,255,0.85)";
        ctx.fillText(
          `${d.description}`,
          startX + 14,
          y + 6
        );
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