import React, { useState } from "react";
import axios from "axios";

export default function AudioUploader({ onAudioData, audioRef }) {
  const [audioUrl, setAudioUrl] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileUpload = async (file) => {
    if (!file) return;

    const url = URL.createObjectURL(file);
    setAudioUrl(url);

    const formData = new FormData();
    formData.append("file", file);

    const res = await axios.post(
      "http://127.0.0.1:8000/process-audio",
      formData,
      { headers: { "Content-Type": "multipart/form-data" } }
    );

    onAudioData?.(res.data);
  };

  return (
    <div>
      {/* Drag & Drop */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          handleFileUpload(e.dataTransfer.files[0]);
        }}
        style={{
          border: "2px dashed #555",
          borderRadius: "12px",
          padding: "2rem",
          textAlign: "center",
          marginBottom: "1rem",
          background: dragActive ? "#141a26" : "#0b0e14",
        }}
      >
        Drop audio here or click to upload! 
        <input
          type="file"
          accept=".wav,.mp3"
          onChange={(e) => handleFileUpload(e.target.files[0])}
          style={{ display: "none" }}
          id="file-input"
        />
        <label htmlFor="file-input" style={{ cursor: "pointer", color: "#8ecae6" }}>
          &nbsp;&nbsp;Browse
        </label>
      </div>

      {/* Audio Player */}
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          controls
          style={{ width: "100%" }}
        />
      )}
    </div>
  );
}
