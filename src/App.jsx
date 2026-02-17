import { useState, useRef, useEffect } from "react";
import VoiceRecorder from "./components/VoiceRecorder";
import MonoVisualizer from "./components/MonoVisualizer";

export default function App() {
  const audioRef = useRef(null);

  const [audioURL, setAudioURL] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [isRecording, setIsRecording] = useState(false);

  async function handleRecordingComplete(blob) {
    setIsRecording(false);
    const url = URL.createObjectURL(blob);
    setAudioURL(url);

    // Send to backend for feature extraction
    const formData = new FormData();
    formData.append("file", blob, "voice.webm");

    const res = await fetch("http://localhost:8000/analyze", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setAnalysisData(data);
  }

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
      <VoiceRecorder
        onRecordingStart={() => setIsRecording(true)}
        onRecordingComplete={handleRecordingComplete}
      />

      {!isRecording && audioURL && (
        <audio
          ref={audioRef}
          controls
          src={audioURL}
          style={{ width: "100%", marginBottom: 20 }}
        />
      )}

      {!isRecording && analysisData && (
        <div style={{ width: "100%", height: "70vh" }}>
          <MonoVisualizer
            audioRef={audioRef}
            data={analysisData}
          />
        </div>
      )}
    </div>
  );
}


  // return (
  //   <div
  //     style={{
  //       width: "100vw",
  //       minHeight: "100vh",
  //       padding: "2rem",
  //       color: "white",
  //       background: "#0b0e14",
  //       boxSizing: "border-box",
  //       overflow: "hidden",
  //     }}
  //   >
  //     {/* Upload + audio player */}
  //     <AudioUploader
  //       audioRef={audioRef}
  //       onFileSelect={(file, url) => {
  //         setAudioFile(file);
  //         setAudioUrl(url);
  //       }}
  //     />

  //     {/* Cluster number control */}
  //     {audioFile && (
  //     <div
  //       style={{
  //         display: "flex",
  //         alignItems: "center",
  //         gap: "12px",
  //         margin: "1rem 0",
  //         padding: "0.5rem 1rem",
  //         borderRadius: "12px",
  //         background: "rgba(255,255,255,0.05)",
  //         backdropFilter: "blur(4px)",
  //         width: "fit-content",
  //       }}
  //     >
  //       <label
  //         htmlFor="cluster-input"
  //         style={{ color: "#8ecae6", fontWeight: 500 }}
  //       >
  //         Sound Groups:
  //       </label>

  //       <input
  //         id="cluster-input"
  //         type="range"
  //         min={2}
  //         max={6}
  //         value={nClusters}
  //         onChange={(e) => setNClusters(Number(e.target.value))}
  //         style={{
  //           accentColor: "#8ecae6",
  //           width: "150px",
  //           cursor: "pointer",
  //         }}
  //       />

  //       <span
  //         style={{
  //           minWidth: "20px",
  //           textAlign: "center",
  //           fontWeight: 600,
  //           color: "#ffffff",
  //         }}
  //       >
  //         {nClusters}
  //       </span>
  //     </div>
  //   )}


  //     {/* Audio Player */}
  //     {audioUrl && (
  //       <audio
  //         ref={audioRef}
  //         src={audioUrl}
  //         controls
  //         style={{ width: "100%", marginBottom: "1rem" }}
  //       />
  //     )}

  //     {/* Visualization */}
  //     {rhythmData && (
  //       <div
  //         style={{
  //           width: "100%",
  //           height: "60vh",
  //           marginTop: "2rem",
  //           position: "relative",
  //         }}
  //       >
  //         <SpiralRhythmVisualizer
  //           audioRef={audioRef}
  //           rhythmData={rhythmData}
  //         />
  //       </div>
  //     )}
  //   </div>
  // );
// }
