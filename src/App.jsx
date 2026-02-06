import React, { useRef, useState, useEffect } from "react";
import AudioUploader from "./components/AudioUploader";
import SpiralRhythmVisualizer from "./components/SpiralRhythmVisualizer";

export default function App() {
  const audioRef = useRef(null);

  const [audioFile, setAudioFile] = useState(null);       // store uploaded file
  const [audioUrl, setAudioUrl] = useState(null);         // for audio player
  const [rhythmData, setRhythmData] = useState(null);     // cluster + onset data
  const [nClusters, setNClusters] = useState(3);          // number of clusters

  // Re-process audio whenever the file or nClusters changes
  useEffect(() => {
    if (!audioFile) return;

    const processAudio = async () => {
      try {
        const formData = new FormData();
        formData.append("file", audioFile);

        const res = await fetch(
          `http://127.0.0.1:8000/process-audio?n_clusters=${nClusters}`,
          {
            method: "POST",
            body: formData,
          }
        );

        const data = await res.json();
        setRhythmData(data);
      } catch (err) {
        console.error("Failed to process audio:", err);
      }
    };

    processAudio();
  }, [audioFile, nClusters]);

  return (
    <div
      style={{
        width: "100vw",
        minHeight: "100vh",
        padding: "2rem",
        color: "white",
        background: "#0b0e14",
        boxSizing: "border-box",
        overflow: "hidden",
      }}
    >
      {/* Upload + audio player */}
      <AudioUploader
        audioRef={audioRef}
        onFileSelect={(file, url) => {
          setAudioFile(file);
          setAudioUrl(url);
        }}
      />

      {/* Cluster number control */}
      {audioFile && (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          margin: "1rem 0",
          padding: "0.5rem 1rem",
          borderRadius: "12px",
          background: "rgba(255,255,255,0.05)",
          backdropFilter: "blur(4px)",
          width: "fit-content",
        }}
      >
        <label
          htmlFor="cluster-input"
          style={{ color: "#8ecae6", fontWeight: 500 }}
        >
          Sound Groups:
        </label>

        <input
          id="cluster-input"
          type="range"
          min={2}
          max={6}
          value={nClusters}
          onChange={(e) => setNClusters(Number(e.target.value))}
          style={{
            accentColor: "#8ecae6",
            width: "150px",
            cursor: "pointer",
          }}
        />

        <span
          style={{
            minWidth: "20px",
            textAlign: "center",
            fontWeight: 600,
            color: "#ffffff",
          }}
        >
          {nClusters}
        </span>
      </div>
    )}


      {/* Audio Player */}
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          controls
          style={{ width: "100%", marginBottom: "1rem" }}
        />
      )}

      {/* Visualization */}
      {rhythmData && (
        <div
          style={{
            width: "100%",
            height: "60vh",
            marginTop: "2rem",
            position: "relative",
          }}
        >
          <SpiralRhythmVisualizer
            audioRef={audioRef}
            rhythmData={rhythmData}
          />
        </div>
      )}
    </div>
  );
}
