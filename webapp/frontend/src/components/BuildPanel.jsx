import React from "react";

export default function BuildPanel({
  models,
  embeddingModel,
  onModelChange,
  onBuild,
  building,
  buildEvents,
  built,
  disabled,
}) {
  const last = buildEvents[buildEvents.length - 1];
  return (
    <div className="panel">
      <div className="panel-title">2. Embeddings</div>

      <div className="field">
        <label>Embedding model</label>
        <select
          value={embeddingModel}
          onChange={(e) => onModelChange(e.target.value)}
          disabled={building}
        >
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        {embeddingModel === "tfidf" && (
          <div className="hint warn">
            TF-IDF builds per-doc vocabularies and is not ideal for the shared
            embed-once store. Prefer openai / jina / minilm.
          </div>
        )}
      </div>

      <div className="hint subtle">
        Chunking strategy is fixed in the backend config (section-aware) and not
        editable here.
      </div>

      <button className="btn primary" onClick={onBuild} disabled={disabled || building}>
        {building ? "Building..." : built ? "Rebuild / Verify" : "Build embeddings"}
      </button>

      {built && !building && <div className="hint ok-text">Embeddings ready for this model.</div>}

      {buildEvents.length > 0 && (
        <div className={`build-progress ${building ? "running" : ""}`}>
          {building && <span className="spin small" />}
          <span className="build-msg">{describe(last)}</span>
        </div>
      )}
    </div>
  );
}

function describe(ev) {
  if (!ev) return "";
  if (ev.stage === "start") return `Starting build (${ev.model})...`;
  if (ev.stage === "chunking" && ev.status === "start") return "Chunking document...";
  if (ev.stage === "chunking" && ev.status === "done") return `Chunked into ${ev.num_chunks} chunks.`;
  if (ev.stage === "embedding" && ev.status === "start")
    return `Embedding ${ev.num_chunks} chunks with ${ev.backend}...`;
  if (ev.stage === "embedding" && ev.status === "done") return `Embedded (${ev.added} added).`;
  if (ev.stage === "done") return ev.message || "Done.";
  if (ev.stage === "error") return `Error: ${ev.message}`;
  return ev.stage;
}
