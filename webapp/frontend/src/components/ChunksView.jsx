import React from "react";

function ChunkCard({ chunk, index, active, onSelect }) {
  return (
    <div
      className={`chunk ${active ? "active" : ""}`}
      onClick={() => onSelect(chunk.id)}
      title="Click to locate in the PDF"
    >
      <div className="chunk-head">
        <span className="chunk-rank">#{index + 1}</span>
        <span className="chunk-score">score {chunk.score}</span>
        <span className="chunk-page">p.{chunk.page}</span>
      </div>
      <div className="chunk-section">{chunk.section_path || chunk.block_type || "(no section)"}</div>
      <div className="chunk-text">{chunk.text}</div>
    </div>
  );
}

export default function ChunksView({ title, chunks, selectedId, onSelect, badge }) {
  if (!chunks || chunks.length === 0) return null;
  return (
    <div className="chunks">
      <div className="chunks-title">
        {title} <span className="count">{chunks.length}</span>
        {badge && <span className="badge subtle">{badge}</span>}
      </div>
      <div className="chunks-list">
        {chunks.map((c, i) => (
          <ChunkCard
            key={`${c.id}-${i}`}
            chunk={c}
            index={i}
            active={selectedId === c.id}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  );
}
