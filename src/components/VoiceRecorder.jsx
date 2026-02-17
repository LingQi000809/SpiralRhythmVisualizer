import { useState, useRef } from "react";

export default function VoiceRecorder({ onRecordingStart, onRecordingComplete }) {
  const [recording, setRecording] = useState(false);

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;

    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    chunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, {
        type: "audio/webm",
      });

      streamRef.current.getTracks().forEach((t) => t.stop());
      onRecordingComplete(blob);
    };

    mediaRecorder.start();
    onRecordingStart(); // notify parent
    setRecording(true);
  }

  function stopRecording() {
    mediaRecorderRef.current.stop();
    setRecording(false);
  }

  return (
    <div style={{ textAlign: "center", marginBottom: 20 }}>
      {!recording ? (
        <button onClick={startRecording} style={buttonStyle}>
          Start Recording
        </button>
      ) : (
        <button
          onClick={stopRecording}
          style={{ ...buttonStyle, background: "#aa3344" }}
        >
          Stop Recording
        </button>
      )}
    </div>
  );
}

const buttonStyle = {
  padding: "12px 24px",
  borderRadius: "8px",
  border: "none",
  background: "#334477",
  color: "white",
  cursor: "pointer",
};
