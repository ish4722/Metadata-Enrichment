import React, { useEffect, useMemo, useState } from "react";
import { api } from "./api.js";
import DocumentPicker from "./components/DocumentPicker.jsx";
import BuildPanel from "./components/BuildPanel.jsx";
import QueryPanel, { CUSTOM } from "./components/QueryPanel.jsx";
import PipelineFlow from "./components/PipelineFlow.jsx";
import ChunksView from "./components/ChunksView.jsx";
import AnswerValidation from "./components/AnswerValidation.jsx";
import PdfViewer from "./components/PdfViewer.jsx";
import Home from "./components/Home.jsx";
import Enrichment from "./components/Enrichment.jsx";
import GroundTruth from "./components/GroundTruth.jsx";

const STAGES_RESET = {
  embed: "pending",
  retrieve: "pending",
  rerank: "pending",
  llm: "pending",
  verify: "pending",
  validate: "pending",
  highlights: "pending",
};

// Floating light/dark switch. Persists to localStorage and toggles the
// `data-theme` attribute on <html>, which flips the CSS variable palette.
function ThemeToggle() {
  const [theme, setTheme] = useState(
    () => document.documentElement.getAttribute("data-theme") || "dark"
  );
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("theme", theme);
    } catch {
      /* ignore storage errors */
    }
  }, [theme]);
  const isLight = theme === "light";
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={() => setTheme(isLight ? "dark" : "light")}
      title={`Switch to ${isLight ? "dark" : "light"} mode`}
      aria-label="Toggle color theme"
    >
      {isLight ? (
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
             strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
             strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="5" />
          <line x1="12" y1="1" x2="12" y2="3" />
          <line x1="12" y1="21" x2="12" y2="23" />
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
          <line x1="1" y1="12" x2="3" y2="12" />
          <line x1="21" y1="12" x2="23" y2="12" />
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
        </svg>
      )}
    </button>
  );
}

export default function App() {
  const [view, setView] = useState("home");
  const [documents, setDocuments] = useState([]);
  const [models, setModels] = useState([]);
  const [queries, setQueries] = useState([]);

  const [selectedDocId, setSelectedDocId] = useState("");
  const [embeddingModel, setEmbeddingModel] = useState("jina-api");

  const [building, setBuilding] = useState(false);
  const [buildEvents, setBuildEvents] = useState([]);

  const [selectedLabel, setSelectedLabel] = useState("");
  const [customQuery, setCustomQuery] = useState("");
  const [retrievalMode, setRetrievalMode] = useState("hybrid");
  const [alpha, setAlpha] = useState(0.5);
  const [useReranker, setUseReranker] = useState(true);
  const [verify, setVerify] = useState(true);

  const [running, setRunning] = useState(false);
  const [stages, setStages] = useState(STAGES_RESET);
  const [retrievedChunks, setRetrievedChunks] = useState([]);
  const [rerankedChunks, setRerankedChunks] = useState([]);
  const [answer, setAnswer] = useState("");
  const [usage, setUsage] = useState(null);
  const [genCost, setGenCost] = useState(null);
  const [embedCost, setEmbedCost] = useState(null);
  const [verifyCost, setVerifyCost] = useState(null);
  const [verifyUsage, setVerifyUsage] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [lastQueryBody, setLastQueryBody] = useState(null);
  const [reverifying, setReverifying] = useState(false);
  const [validation, setValidation] = useState(null);
  const [highlightPages, setHighlightPages] = useState([]);
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [error, setError] = useState("");

  const selectedDoc = documents.find((d) => d.id === selectedDocId);
  const built = !!selectedDoc?.built_models?.includes(embeddingModel);

  async function refreshDocuments() {
    const { documents } = await api.documents();
    setDocuments(documents);
  }

  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    if (view !== "explorer" || loaded) return;
    (async () => {
      try {
        await refreshDocuments();
        const m = await api.embeddingModels();
        setModels(m.models);
        setEmbeddingModel(m.default || m.models[0]);
        const q = await api.queries();
        setQueries(q.queries);
        setLoaded(true);
      } catch (e) {
        setError(`Failed to reach backend: ${e.message}`);
      }
    })();
  }, [view, loaded]);

  async function handleBuild() {
    if (!selectedDocId) return;
    setBuilding(true);
    setBuildEvents([]);
    setError("");
    try {
      await api.stream("/build", { doc_id: selectedDocId, embedding_model: embeddingModel }, (ev) => {
        setBuildEvents((prev) => [...prev, ev]);
        if (ev.stage === "error") setError(ev.message);
      });
      await refreshDocuments();
    } catch (e) {
      setError(`Build failed: ${e.message}`);
    } finally {
      setBuilding(false);
    }
  }

  function resetQueryState() {
    setStages({ ...STAGES_RESET });
    setRetrievedChunks([]);
    setRerankedChunks([]);
    setAnswer("");
    setUsage(null);
    setGenCost(null);
    setEmbedCost(null);
    setVerifyCost(null);
    setVerifyUsage(null);
    setConfidence(null);
    setValidation(null);
    setHighlightPages([]);
    setSelectedChunkId(null);
    setError("");
  }

  function buildQueryBody() {
    const base = {
      doc_id: selectedDocId,
      embedding_model: embeddingModel,
      retrieval_mode: retrievalMode,
      hybrid_alpha: alpha,
      use_reranker: useReranker,
      verify: verify,
    };
    if (selectedLabel === CUSTOM) {
      return { ...base, question: customQuery, retrieval_query: customQuery };
    }
    const q = queries.find((x) => x.label === selectedLabel);
    if (!q) return null;
    return { ...base, label: q.label, question: q.question, retrieval_query: q.retrieval_query };
  }

  function setStage(id, status) {
    setStages((prev) => ({ ...prev, [id]: status }));
  }

  function applyVerifyEvent(ev) {
    if (ev.status === "start") {
      setStage("verify", "active");
      setConfidence({ running: true, planning: true, confidence: null, band: null,
        checks: [], total: ev.num_questions ?? null, currentIndex: null });
    } else if (ev.status === "planned") {
      setStage("verify", "active");
      setConfidence({
        running: true, planning: false, confidence: null, band: null,
        total: ev.count, currentIndex: null,
        checks: (ev.checks || []).map((c) => ({ ...c, verdict: null })),
      });
    } else if (ev.status === "checking") {
      setConfidence((prev) => (prev ? { ...prev, currentIndex: ev.index } : prev));
      setStage("verify", "active");
    } else if (ev.status === "check_done") {
      setConfidence((prev) => {
        if (!prev) return prev;
        const checks = [...(prev.checks || [])];
        checks[ev.index] = ev.check;
        return { ...prev, checks };
      });
    } else if (ev.status === "done") {
      setStage("verify", "done");
      setVerifyCost(ev.cost_usd ?? null);
      setVerifyUsage(ev.usage ?? null);
      setConfidence({
        running: false,
        confidence: ev.confidence,
        band: ev.band,
        checks: ev.checks || [],
        breakdown: ev.breakdown || null,
        verbalized_confidence: ev.verbalized_confidence,
        suggestion: ev.suggestion || null,
      });
    } else if (ev.status === "skipped") {
      setStage("verify", "skipped");
    } else if (ev.status === "error") {
      setStage("verify", "error");
    } else {
      setStage("verify", "active");
    }
  }

  async function handleAcceptSuggestion(suggested) {
    if (!lastQueryBody || !suggested) return;
    setReverifying(true);
    setError("");
    // The accepted value becomes the displayed answer, then we re-verify it.
    setAnswer(suggested);
    setStage("verify", "active");
    setConfidence({ running: true, planning: true, confidence: null, band: null,
      checks: [], total: null, currentIndex: null });
    try {
      await api.stream("/reverify", { ...lastQueryBody, answer: suggested }, (ev) => {
        if (ev.stage === "verify") applyVerifyEvent(ev);
        else if (ev.stage === "error") setError(ev.message);
      });
    } catch (e) {
      setError(`Re-verify failed: ${e.message}`);
    } finally {
      setReverifying(false);
    }
  }

  function handleRejectSuggestion() {
    setConfidence((prev) => (prev ? { ...prev, suggestion: null } : prev));
  }

  async function handleRun() {
    const body = buildQueryBody();
    if (!body || (!body.question && !body.label)) {
      setError("Pick a query or type a custom one.");
      return;
    }
    resetQueryState();
    setLastQueryBody(body);
    setRunning(true);
    if (!useReranker) setStage("rerank", "skipped");
    if (!verify) setStage("verify", "skipped");

    try {
      await api.stream("/query", body, (ev) => {
        switch (ev.stage) {
          case "embed":
            setStage("embed", ev.status === "done" ? "done" : "active");
            if (ev.status === "done" && ev.embedding_cost_usd != null) {
              setEmbedCost(ev.embedding_cost_usd);
            }
            break;
          case "retrieve":
            if (ev.status === "done") {
              setStage("retrieve", "done");
              setRetrievedChunks(ev.chunks || []);
            } else setStage("retrieve", "active");
            break;
          case "rerank":
            if (ev.status === "done") {
              setStage("rerank", "done");
              setRerankedChunks(ev.chunks || []);
            } else if (ev.status === "skipped" || ev.status === "unavailable") {
              setStage("rerank", "skipped");
            } else setStage("rerank", "active");
            break;
          case "llm":
            if (ev.status === "done") {
              setStage("llm", "done");
              setAnswer(ev.answer || "");
              setUsage(ev.usage || null);
              setGenCost(ev.cost_usd ?? null);
            } else if (ev.status === "skipped") {
              setStage("llm", "skipped");
            } else setStage("llm", "active");
            break;
          case "verify":
            applyVerifyEvent(ev);
            break;
          case "validate":
            if (ev.status === "skipped") {
              setStage("validate", "skipped");
              setValidation({ skipped: true, ground_truth: ev.ground_truth });
            } else if (ev.status === "done") {
              setStage("validate", "done");
              setValidation({
                skipped: false,
                ground_truth: ev.ground_truth,
                retrieval_hit: ev.retrieval_hit,
                token_match: ev.token_match,
                judge_match: ev.judge_match,
              });
            }
            break;
          case "highlights":
            setStage("highlights", ev.status === "done" ? "done" : "error");
            if (ev.status === "done") setHighlightPages(ev.pages || []);
            break;
          case "error":
            setError(ev.message);
            break;
          default:
            break;
        }
      });
    } catch (e) {
      setError(`Query failed: ${e.message}`);
    } finally {
      setRunning(false);
    }
  }

  const finalChunks = rerankedChunks.length > 0 ? rerankedChunks : retrievedChunks;

  const canQuery = useMemo(
    () => !!selectedDocId && built && !running && !building,
    [selectedDocId, built, running, building]
  );

  if (view === "home") {
    return (
      <>
        <ThemeToggle />
        <Home
          onOpenExplorer={() => setView("explorer")}
          onOpenEnrichment={() => setView("enrichment")}
          onOpenGroundTruth={() => setView("groundtruth")}
        />
      </>
    );
  }

  if (view === "enrichment") {
    return (
      <>
        <ThemeToggle />
        <Enrichment onHome={() => setView("home")} />
      </>
    );
  }

  if (view === "groundtruth") {
    return (
      <>
        <ThemeToggle />
        <GroundTruth onHome={() => setView("home")} />
      </>
    );
  }

  return (
    <div className="app">
      <ThemeToggle />
      <header className="topbar">
        <button type="button" className="home-back" onClick={() => setView("home")}>
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
               strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="19" y1="12" x2="5" y2="12" />
            <polyline points="12 19 5 12 12 5" />
          </svg>
          Home
        </button>
        <h1>RAG Explorer</h1>
        <span className="subtle">
          {"PDF -> datalab -> embeddings -> retrieve -> rerank -> answer -> validate"}
        </span>
      </header>

      {error && <div className="error-bar">{error}</div>}

      <div className="layout">
        <aside className="sidebar">
          <DocumentPicker
            documents={documents}
            selectedId={selectedDocId}
            onSelect={(id) => {
              setSelectedDocId(id);
              resetQueryState();
              setBuildEvents([]);
            }}
            onRefresh={refreshDocuments}
          />
          <BuildPanel
            models={models}
            embeddingModel={embeddingModel}
            onModelChange={setEmbeddingModel}
            onBuild={handleBuild}
            building={building}
            buildEvents={buildEvents}
            built={built}
            disabled={!selectedDocId || !selectedDoc?.has_datalab}
          />
          <QueryPanel
            queries={queries}
            selectedLabel={selectedLabel}
            onSelectLabel={setSelectedLabel}
            customQuery={customQuery}
            onCustomChange={setCustomQuery}
            retrievalMode={retrievalMode}
            onRetrievalMode={setRetrievalMode}
            alpha={alpha}
            onAlpha={setAlpha}
            useReranker={useReranker}
            onReranker={setUseReranker}
            verify={verify}
            onVerify={setVerify}
            onRun={handleRun}
            running={running}
            disabled={!canQuery}
          />
          {selectedDocId && !built && (
            <div className="hint warn panel-hint">
              Build embeddings for "{embeddingModel}" before querying.
            </div>
          )}
        </aside>

        <main className="main">
          <section className="card flow-card">
            <PipelineFlow stages={stages} />
          </section>

          <div className="results">
            <div className="results-left">
              <AnswerValidation
                answer={answer}
                usage={usage}
                genCost={genCost}
                confidence={confidence}
                validation={validation}
                reverifying={reverifying}
                onAcceptSuggestion={handleAcceptSuggestion}
                onRejectSuggestion={handleRejectSuggestion}
              />
              <ChunksView
                title="Retrieved chunks"
                chunks={retrievedChunks}
                selectedId={selectedChunkId}
                onSelect={setSelectedChunkId}
                badge={retrievalMode === "hybrid" ? `hybrid a=${alpha.toFixed(2)}` : "dense"}
              />
              {rerankedChunks.length > 0 && (
                <ChunksView
                  title="Reranked chunks"
                  chunks={rerankedChunks}
                  selectedId={selectedChunkId}
                  onSelect={setSelectedChunkId}
                  badge="cross-encoder"
                />
              )}
            </div>

            <div className="results-right">
              <PdfViewer
                docId={selectedDoc?.has_pdf ? selectedDocId : ""}
                highlightPages={highlightPages}
                selectedChunkId={selectedChunkId}
              />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
