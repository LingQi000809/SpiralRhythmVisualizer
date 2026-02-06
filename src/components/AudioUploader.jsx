import React, { useState } from "react";

export default function AudioUploader({ onFileSelect, audioRef }) {
  const [dragActive, setDragActive] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);

  const handleFileUpload = (file) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setAudioUrl(url);

    // notify parent about the selected file
    onFileSelect?.(file, url);
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
          cursor: "pointer",
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
        <label
          htmlFor="file-input"
          style={{ cursor: "pointer", color: "#8ecae6", marginLeft: "8px" }}
        >
          Browse
        </label>
      </div>
    </div>
  );
}
