import React, { useEffect, useMemo, useState } from "react";
import { api } from "./api.js";
import DocumentPicker from "./components/DocumentPicker.jsx";
import BuildPanel from "./components/BuildPanel.jsx";
import QueryPanel, { CUSTOM } from "./components/QueryPanel.jsx";
import PipelineFlow from "./components/PipelineFlow.jsx";
import ChunksView from "./components/ChunksView.jsx";
import AnswerValidation from "./components/AnswerValidation.jsx";
import PdfViewer from "./components/PdfViewer.jsx";

const STAGES_RESET = {
  embed: "pending",
  retrieve: "pending",
  rerank: "pending",
  llm: "pending",
  validate: "pending",
  highlights: "pending",
};

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [models, setModels] = useState([]);
  const [queries, setQueries] = useState([]);

  const [selectedDocId, setSelectedDocId] = useState("");
  const [embeddingModel, setEmbeddingModel] = useState("openai");

  const [building, setBuilding] = useState(false);
  const [buildEvents, setBuildEvents] = useState([]);

  const [selectedLabel, setSelectedLabel] = useState("");
  const [customQuery, setCustomQuery] = useState("");
  const [retrievalMode, setRetrievalMode] = useState("hybrid");
  const [alpha, setAlpha] = useState(0.5);
  const [useReranker, setUseReranker] = useState(true);

  const [running, setRunning] = useState(false);
  const [stages, setStages] = useState(STAGES_RESET);
  const [retrievedChunks, setRetrievedChunks] = useState([]);
  const [rerankedChunks, setRerankedChunks] = useState([]);
  const [answer, setAnswer] = useState("");
  const [usage, setUsage] = useState(null);
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

  useEffect(() => {
    (async () => {
      try {
        await refreshDocuments();
        const m = await api.embeddingModels();
        setModels(m.models);
        setEmbeddingModel(m.default || m.models[0]);
        const q = await api.queries();
        setQueries(q.queries);
      } catch (e) {
        setError(`Failed to reach backend: ${e.message}`);
      }
    })();
  }, []);

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

  async function handleRun() {
    const body = buildQueryBody();
    if (!body || (!body.question && !body.label)) {
      setError("Pick a query or type a custom one.");
      return;
    }
    resetQueryState();
    setRunning(true);
    if (!useReranker) setStage("rerank", "skipped");

    try {
      await api.stream("/query", body, (ev) => {
        switch (ev.stage) {
          case "embed":
            setStage("embed", ev.status === "done" ? "done" : "active");
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
            } else if (ev.status === "skipped") {
              setStage("llm", "skipped");
            } else setStage("llm", "active");
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

  return (
    <div className="app">
      <header className="topbar">
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
              <AnswerValidation answer={answer} usage={usage} validation={validation} />
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
