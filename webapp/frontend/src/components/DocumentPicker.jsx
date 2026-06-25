import React, { useRef, useState } from "react";
import { api } from "../api.js";

function StatusDot({ ok, label }) {
  return (
    <span className={`statusdot ${ok ? "ok" : "missing"}`} title={label}>
      {label}
    </span>
  );
}

export default function DocumentPicker({ documents, selectedId, onSelect, onRefresh }) {
  const fileRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [msg, setMsg] = useState("");

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setMsg("");
    try {
      const info = await api.upload(file);
      await onRefresh?.();
      onSelect(info.id);
      const parts = [];
      parts.push(info.has_datalab ? "datalab matched" : "NO datalab match");
      parts.push(info.has_matched ? "ground truth found" : "no ground truth");
      setMsg(`Uploaded as ${info.uploaded_as} -> id ${info.id} (${parts.join(", ")})`);
    } catch (err) {
      setMsg(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  const selected = documents.find((d) => d.id === selectedId);

  return (
    <div className="panel">
      <div className="panel-title">1. Document</div>

      <label className="upload-btn">
        <input
          ref={fileRef}
          type="file"
          accept="application/pdf"
          onChange={handleUpload}
          disabled={uploading}
          hidden
        />
        {uploading ? "Uploading..." : "Upload a PDF"}
      </label>

      <div className="field">
        <label>Or pick an existing document</label>
        <select value={selectedId || ""} onChange={(e) => onSelect(e.target.value)}>
          <option value="">-- select --</option>
          {documents.map((d) => (
            <option key={d.id} value={d.id}>
              {d.id}
              {d.has_pdf ? "" : " (no pdf)"}
            </option>
          ))}
        </select>
      </div>

      {msg && <div className="hint">{msg}</div>}

      {selected && (
        <div className="doc-status">
          <StatusDot ok={selected.has_pdf} label="PDF" />
          <StatusDot ok={selected.has_datalab} label="datalab" />
          <StatusDot ok={selected.has_matched} label="ground truth" />
          {selected.built_models?.length > 0 && (
            <div className="hint">built: {selected.built_models.join(", ")}</div>
          )}
        </div>
      )}
    </div>
  );
}
