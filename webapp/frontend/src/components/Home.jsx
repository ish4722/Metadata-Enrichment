import React from "react";

// Gold-set reviewer and the analyzer are served outside the Vite dev server, so
// we link to them with absolute URLs. The reviewer lives on the FastAPI backend
// (port 8000); the analyzer is a fully standalone page served by Vite from
// /public (works even when the backend is down).
const BACKEND_ORIGIN = `${window.location.protocol}//${window.location.hostname}:8000`;

const TOOLS = [
  {
    id: "explorer",
    title: "RAG Explorer",
    tagline: "Run the full pipeline, end to end",
    desc:
      "Pick a document, build embeddings, query it, and watch the retrieve -> rerank -> answer -> validate flow with retrieved chunks highlighted on the PDF.",
    chips: ["embeddings", "hybrid + rerank", "PDF highlights"],
    accent: "var(--accent)",
    needsBackend: true,
    icon: (
      <>
        <circle cx="11" cy="11" r="7" />
        <line x1="16.5" y1="16.5" x2="21" y2="21" />
      </>
    ),
  },
  {
    id: "goldset",
    title: "Gold-Set Reviewer",
    tagline: "Inspect & curate relevance labels",
    desc:
      "Browse the graded retrieval gold set per document, see correct vs wrong chunks on the PDF, and add or remove relevance labels by hand.",
    chips: ["graded labels", "per-document", "edit & save"],
    accent: "var(--accent-2)",
    needsBackend: true,
    href: `${BACKEND_ORIGIN}/review`,
    icon: (
      <>
        <path d="M4 5a2 2 0 0 1 2-2h9l5 5v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" />
        <polyline points="14 3 14 8 19 8" />
        <polyline points="9 13 11 15 15 11" />
      </>
    ),
  },
  {
    id: "analyzer",
    title: "JSON Field Analyzer",
    tagline: "Drop a JSON, inspect its fields",
    desc:
      "A fully client-side tool: drop any JSON file to explore its structure, field coverage and values. No server required — works completely offline.",
    chips: ["client-side", "no backend", "drag & drop"],
    accent: "var(--warn)",
    needsBackend: false,
    href: "/json_field_analyzer.html",
    icon: (
      <>
        <path d="M8 3H6a2 2 0 0 0-2 2v4a2 2 0 0 1-2 2 2 2 0 0 1 2 2v4a2 2 0 0 0 2 2h2" />
        <path d="M16 3h2a2 2 0 0 1 2 2v4a2 2 0 0 0 2 2 2 2 0 0 0-2 2v4a2 2 0 0 1-2 2h-2" />
      </>
    ),
  },
];

export default function Home({ onOpenExplorer }) {
  function handleOpen(tool) {
    if (tool.id === "explorer") {
      onOpenExplorer();
    } else {
      window.open(tool.href, "_blank", "noopener");
    }
  }

  return (
    <div className="home">
      <header className="home-hero">
        <div className="home-badge">Metadata Enrichment</div>
        <h1>RAG Workbench</h1>
        <p>
          A unified workspace for retrieval-augmented metadata extraction. Choose a tool to get
          started.
        </p>
      </header>

      <div className="home-grid">
        {TOOLS.map((tool) => (
          <button
            key={tool.id}
            type="button"
            className="home-card"
            style={{ "--card-accent": tool.accent }}
            onClick={() => handleOpen(tool)}
          >
            <div className="home-card-top">
              <span className="home-card-icon" aria-hidden="true">
                <svg
                  viewBox="0 0 24 24"
                  width="26"
                  height="26"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  {tool.icon}
                </svg>
              </span>
              <span
                className={`home-card-tag ${tool.needsBackend ? "needs" : "offline"}`}
              >
                {tool.needsBackend ? "needs backend" : "works offline"}
              </span>
            </div>

            <h2>{tool.title}</h2>
            <div className="home-card-tagline">{tool.tagline}</div>
            <p className="home-card-desc">{tool.desc}</p>

            <div className="home-card-chips">
              {tool.chips.map((c) => (
                <span key={c} className="home-chip">
                  {c}
                </span>
              ))}
            </div>

            <span className="home-card-cta">
              Open
              <svg
                viewBox="0 0 24 24"
                width="16"
                height="16"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="5" y1="12" x2="19" y2="12" />
                <polyline points="12 5 19 12 12 19" />
              </svg>
            </span>
          </button>
        ))}
      </div>

      <footer className="home-foot">
        <span className="subtle">
          Tip: the Gold-Set Reviewer and RAG Explorer need the backend running on port 8000; the
          JSON Field Analyzer runs entirely in your browser.
        </span>
      </footer>
    </div>
  );
}
