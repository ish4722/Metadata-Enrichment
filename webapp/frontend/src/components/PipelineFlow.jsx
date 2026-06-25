import React from "react";

const STAGES = [
  { id: "embed", label: "Embed query", icon: "</>" },
  { id: "retrieve", label: "Retrieve", icon: "search" },
  { id: "rerank", label: "Rerank", icon: "sort" },
  { id: "llm", label: "LLM answer", icon: "AI" },
  { id: "validate", label: "Validate", icon: "check" },
  { id: "highlights", label: "Highlight PDF", icon: "mark" },
];

// status per stage: "pending" | "active" | "done" | "skipped" | "error"
export default function PipelineFlow({ stages }) {
  return (
    <div className="flow">
      {STAGES.map((s, i) => {
        const status = stages[s.id] || "pending";
        return (
          <React.Fragment key={s.id}>
            <div className={`flow-node ${status}`}>
              <div className="flow-dot">
                {status === "done" && <span className="tick">&#10003;</span>}
                {status === "active" && <span className="spin" />}
                {status === "skipped" && <span className="muted">&minus;</span>}
                {status === "error" && <span className="err">!</span>}
              </div>
              <div className="flow-label">{s.label}</div>
            </div>
            {i < STAGES.length - 1 && (
              <div className={`flow-edge ${status === "done" ? "filled" : ""}`}>
                <span className="pulse" />
              </div>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
