import React from "react";

export const CUSTOM = "__custom__";

export default function QueryPanel({
  queries,
  selectedLabel,
  onSelectLabel,
  customQuery,
  onCustomChange,
  retrievalMode,
  onRetrievalMode,
  alpha,
  onAlpha,
  useReranker,
  onReranker,
  onRun,
  running,
  disabled,
}) {
  const selected = queries.find((q) => q.label === selectedLabel);
  const isCustom = selectedLabel === CUSTOM;

  return (
    <div className="panel">
      <div className="panel-title">3. Query</div>

      <div className="field">
        <label>Query</label>
        <select value={selectedLabel || ""} onChange={(e) => onSelectLabel(e.target.value)}>
          <option value="">-- select a query --</option>
          {queries.map((q) => (
            <option key={q.label} value={q.label}>
              {q.label}
              {q.validatable ? " (validatable)" : ""}
            </option>
          ))}
          <option value={CUSTOM}>Custom query...</option>
        </select>
      </div>

      {isCustom ? (
        <div className="field">
          <label>Custom question (no validation)</label>
          <textarea
            rows={3}
            value={customQuery}
            placeholder="Ask anything about this document..."
            onChange={(e) => onCustomChange(e.target.value)}
          />
        </div>
      ) : (
        selected && <div className="query-preview">{selected.question}</div>
      )}

      <div className="field">
        <label>Retrieval mode</label>
        <div className="seg">
          <button
            className={retrievalMode === "dense" ? "on" : ""}
            onClick={() => onRetrievalMode("dense")}
          >
            Dense
          </button>
          <button
            className={retrievalMode === "hybrid" ? "on" : ""}
            onClick={() => onRetrievalMode("hybrid")}
          >
            Hybrid
          </button>
        </div>
      </div>

      <div className={`field ${retrievalMode === "hybrid" ? "" : "disabled"}`}>
        <label>
          Hybrid alpha: <b>{alpha.toFixed(2)}</b>{" "}
          <span className="subtle">(0 = BM25 / keyword, 1 = dense)</span>
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={alpha}
          disabled={retrievalMode !== "hybrid"}
          onChange={(e) => onAlpha(parseFloat(e.target.value))}
        />
      </div>

      <div className="field toggle-row">
        <label className="switch">
          <input type="checkbox" checked={useReranker} onChange={(e) => onReranker(e.target.checked)} />
          <span className="slider" />
        </label>
        <span>Cross-encoder reranker {useReranker ? "ON" : "OFF"}</span>
      </div>

      <button className="btn primary" onClick={onRun} disabled={disabled || running}>
        {running ? "Running..." : "Run query"}
      </button>
    </div>
  );
}
