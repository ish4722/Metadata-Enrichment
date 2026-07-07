import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import ChunksView from "./ChunksView.jsx";
import PdfViewer from "./PdfViewer.jsx";

function StatusDot({ ok, label }) {
  return (
    <span className={`statusdot ${ok ? "ok" : "missing"}`} title={label}>
      {label}
    </span>
  );
}

// ── process timeline ──────────────────────────────────────────────────────────
const TIMELINE = [
  { id: "embed", label: "PDF embedded", hint: "Chunk + embed the document into the vector store" },
  { id: "fields", label: "Fields read", hint: "Attributes extracted from the uploaded JSON" },
  { id: "genq", label: "Generate queries", hint: "LLM writes one retrieval query per field" },
  { id: "retrieve", label: "Retrieval", hint: "Find the most relevant chunks per attribute" },
  { id: "rerank", label: "Reranking", hint: "Cross-encoder re-scores the shortlist" },
  { id: "generate", label: "Answer", hint: "LLM answers each attribute" },
  { id: "output", label: "Output", hint: "Enriched JSON assembled (same file, filled)" },
];

function ProcessTimeline({ phases, answered, total, genqDone, onBlock, activeBlock }) {
  return (
    <div className="flow">
      {TIMELINE.map((s, i) => {
        const status = phases[s.id] || "pending";
        const showCount =
          (s.id === "generate" && total > 0) || (s.id === "genq" && total > 0);
        const countText = s.id === "genq" ? `${genqDone}/${total}` : `${answered}/${total}`;
        return (
          <React.Fragment key={s.id}>
            <div
              className={`flow-node ${status} ${activeBlock === s.id ? "picked" : ""}`}
              onClick={() => onBlock(s.id)}
              title={s.hint}
            >
              <div className="flow-dot">
                {status === "done" && <span className="tick">&#10003;</span>}
                {status === "active" && <span className="spin" />}
                {status === "skipped" && <span className="muted">&minus;</span>}
                {status === "error" && <span className="err">!</span>}
              </div>
              <div className="flow-label">{s.label}</div>
              {showCount && <div className="flow-sub subtle">{countText}</div>}
            </div>
            {i < TIMELINE.length - 1 && (
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

const LOOP_STEPS = [
  { id: "retrieve", label: "Retrieve" },
  { id: "rerank", label: "Rerank" },
  { id: "generate", label: "Generate" },
];

function LoopProgress({ running, answered, total, currentLabel, activeSub, cost }) {
  const pctDone = total ? Math.round((answered / total) * 100) : 0;
  const totalCost = (cost.embedding || 0) + (cost.generation || 0);
  const position = currentLabel && total ? Math.min(answered + 1, total) : answered;

  return (
    <div className="loop-wrap">
      <div className="loop-lane">
        <div className={`loop-ring ${running ? "spinning" : ""}`} aria-hidden="true">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
               strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
        </div>

        <div className="loop-steps">
          {LOOP_STEPS.map((s, i) => (
            <React.Fragment key={s.id}>
              <span className={`loop-chip ${running && activeSub === s.id ? "on" : ""}`}>
                {s.label}
              </span>
              {i < LOOP_STEPS.length - 1 && <span className="loop-arrow">&rarr;</span>}
            </React.Fragment>
          ))}
          <span className="loop-back" title="repeats for every attribute">&#8630; per query</span>
        </div>

        <div className="loop-status subtle">
          {running && currentLabel
            ? `Attribute ${position}/${total}: ${currentLabel}`
            : total
            ? `${answered}/${total} attributes done`
            : "Waiting to start"}
        </div>
      </div>

      <div className="loop-track">
        <div className="loop-fill" style={{ width: `${pctDone}%` }}>
          <span className="loop-marker" />
        </div>
      </div>

      <div className="loop-foot">
        <span className="subtle">{pctDone}% complete</span>
        <span className="cost-chip" title="Estimated LLM + embedding cost for this run">
          <span className="cost-dot" /> ${totalCost.toFixed(4)}
        </span>
      </div>
    </div>
  );
}

// ── categories / verdicts ─────────────────────────────────────────────────────
const CATEGORY_LABEL = {
  validated_correct: "validated",
  validated_incorrect: "mismatch",
  enriched_confident: "enriched",
  enriched_low_conf: "low confidence",
  not_found: "not found",
};

function effectiveCategory(attr, verdict) {
  if (verdict === "correct") return "validated_correct";
  if (verdict === "incorrect") return "validated_incorrect";
  return attr.category;
}

function toneForCategory(cat) {
  switch (cat) {
    case "validated_correct":
      return "good";
    case "validated_incorrect":
      return "bad";
    case "enriched_confident":
      return "enriched";
    case "enriched_low_conf":
      return "warn";
    default:
      return "muted";
  }
}

// ── dashboard ─────────────────────────────────────────────────────────────────
function pct(n, d) {
  if (!d) return "0%";
  return `${Math.round((n / d) * 100)}%`;
}

function Dashboard({ attributes, humanVerdicts }) {
  const done = attributes.filter((a) => a.status === "done");
  const total = done.length;
  const counts = {
    validated_correct: 0,
    validated_incorrect: 0,
    enriched_confident: 0,
    enriched_low_conf: 0,
    not_found: 0,
  };
  done.forEach((a) => {
    const cat = effectiveCategory(a, humanVerdicts[a.label]);
    if (counts[cat] !== undefined) counts[cat] += 1;
  });
  const humanValidated = Object.keys(humanVerdicts).length;
  const enriched = counts.enriched_confident + counts.enriched_low_conf;

  const cards = [
    { key: "vc", label: "Validated correct", value: pct(counts.validated_correct, total), sub: `${counts.validated_correct} of ${total}`, tone: "good" },
    { key: "vi", label: "Validated incorrect", value: pct(counts.validated_incorrect, total), sub: `${counts.validated_incorrect} of ${total}`, tone: "bad" },
    { key: "hv", label: "Human verified", value: `${humanValidated}`, sub: "reviewed by you", tone: "info" },
    { key: "en", label: "Enriched (null field)", value: `${enriched}`, sub: "filled from document", tone: "info" },
    { key: "ec", label: "Confident enriched", value: `${counts.enriched_confident}`, sub: "high confidence", tone: "good" },
    { key: "el", label: "Low-confidence enriched", value: `${counts.enriched_low_conf}`, sub: "needs review", tone: "warn" },
    { key: "nf", label: "Not found", value: `${counts.not_found}`, sub: "kept null", tone: "muted" },
  ];

  return (
    <div className="enrich-dash">
      {cards.map((c) => (
        <div key={c.key} className={`dash-card ${c.tone}`}>
          <div className="dash-value">{c.value}</div>
          <div className="dash-label">{c.label}</div>
          <div className="dash-sub subtle">{c.sub}</div>
        </div>
      ))}
    </div>
  );
}

function ConfidenceBar({ value }) {
  if (value === null || value === undefined) return <span className="subtle">n/a</span>;
  const p = Math.round(value * 100);
  const tone = value >= 0.6 ? "good" : value >= 0.35 ? "warn" : "bad";
  return (
    <div className="conf-bar" title={`confidence ${p}%`}>
      <div className={`conf-fill ${tone}`} style={{ width: `${p}%` }} />
      <span className="conf-num">{p}%</span>
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────
const PHASES_RESET = {
  embed: "pending",
  fields: "pending",
  genq: "pending",
  retrieve: "pending",
  rerank: "pending",
  generate: "pending",
  output: "pending",
};

export default function Enrichment({ onHome }) {
  const [documents, setDocuments] = useState([]);
  const [loaded, setLoaded] = useState(false);

  const [selectedDocId, setSelectedDocId] = useState("");
  const fileRef = useRef(null);
  const jsonRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const [structured, setStructured] = useState(null);
  const [structuredName, setStructuredName] = useState("");
  const [structuredCount, setStructuredCount] = useState(0);
  const [structuredSource, setStructuredSource] = useState(null); // "auto" | "manual"
  const [jsonMsg, setJsonMsg] = useState("");

  const [running, setRunning] = useState(false);
  const [phases, setPhases] = useState(PHASES_RESET);
  const [buildMsg, setBuildMsg] = useState("");
  const [queriesInfo, setQueriesInfo] = useState(null);
  const [genqDone, setGenqDone] = useState(0);
  const [attributes, setAttributes] = useState([]);
  const [currentLabel, setCurrentLabel] = useState(null);
  const [activeSub, setActiveSub] = useState(null);
  const [cost, setCost] = useState({ embedding: 0, generation: 0 });
  const [enrichedJson, setEnrichedJson] = useState(null);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");

  const [selectedLabel, setSelectedLabel] = useState(null);
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [humanVerdicts, setHumanVerdicts] = useState({});
  const [detailBlock, setDetailBlock] = useState(null);

  const selectedDoc = documents.find((d) => d.id === selectedDocId);

  async function refreshDocuments() {
    const { documents } = await api.enrichmentDocuments();
    setDocuments(documents);
    return documents;
  }

  useEffect(() => {
    if (loaded) return;
    (async () => {
      try {
        await refreshDocuments();
        setLoaded(true);
      } catch (e) {
        setError(`Failed to reach backend: ${e.message}`);
      }
    })();
  }, [loaded]);

  function clearStructured() {
    setStructured(null);
    setStructuredName("");
    setStructuredCount(0);
    setStructuredSource(null);
    setJsonMsg("");
  }

  // Pull the structured JSON already on disk for a doc so the user does not have
  // to upload one when the PDF already has a matching structured file.
  async function loadMatchedJson(docId) {
    try {
      const record = await api.matched(docId);
      const data =
        record && typeof record.data === "object" && record.data ? record.data : record;
      const count = data && typeof data === "object" ? Object.keys(data).length : 0;
      setStructured(record);
      setStructuredName(`${docId}.json`);
      setStructuredCount(count);
      setStructuredSource("auto");
      setJsonMsg(`Using the structured JSON already on file for ${docId} (${count} fields).`);
      return true;
    } catch (err) {
      return false;
    }
  }

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg("");
    try {
      const info = await api.upload(file);
      await refreshDocuments();
      setSelectedDocId(info.id);
      resetRun();
      clearStructured();
      // If a matching structured JSON already exists, load it automatically so
      // the user is not asked to upload one.
      if (info.has_matched) {
        await loadMatchedJson(info.id);
      }
      const parts = [
        info.has_datalab ? "datalab matched" : "NO datalab match",
        info.has_matched ? "structured file auto-loaded" : "no structured file",
      ];
      setUploadMsg(`Uploaded ${info.uploaded_as} -> ${info.id} (${parts.join(", ")})`);
    } catch (err) {
      setUploadMsg(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  function handleJsonUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setJsonMsg("");
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result));
        const data = parsed && typeof parsed.data === "object" && parsed.data ? parsed.data : parsed;
        const count = data && typeof data === "object" ? Object.keys(data).length : 0;
        setStructured(parsed);
        setStructuredName(file.name);
        setStructuredCount(count);
        setStructuredSource("manual");
        setJsonMsg(`Loaded ${file.name} (${count} fields)`);
        resetRun();
      } catch (err) {
        setStructured(null);
        setStructuredName("");
        setStructuredCount(0);
        setStructuredSource(null);
        setJsonMsg(`Invalid JSON: ${err.message}`);
      }
    };
    reader.onerror = () => setJsonMsg("Could not read the file.");
    reader.readAsText(file);
    if (jsonRef.current) jsonRef.current.value = "";
  }

  function setPhase(id, status) {
    setPhases((prev) => ({ ...prev, [id]: status }));
  }

  function resetRun() {
    setPhases({ ...PHASES_RESET });
    setBuildMsg("");
    setQueriesInfo(null);
    setGenqDone(0);
    setAttributes([]);
    setCurrentLabel(null);
    setActiveSub(null);
    setCost({ embedding: 0, generation: 0 });
    setEnrichedJson(null);
    setSummary(null);
    setSelectedLabel(null);
    setSelectedChunkId(null);
    setHumanVerdicts({});
    setDetailBlock(null);
    setError("");
  }

  function markSubStage(sub) {
    setPhases((prev) => {
      const order = ["retrieve", "rerank", "generate"];
      const cur = order.indexOf(sub);
      const next = { ...prev };
      order.forEach((id, i) => {
        if (next[id] === "done") return;
        next[id] = i < cur ? "done" : i === cur ? "active" : "pending";
      });
      return next;
    });
  }

  function updateAttr(index, patch) {
    setAttributes((prev) => {
      const next = [...prev];
      const existing = next[index] || { index };
      next[index] = { ...existing, ...patch };
      return next;
    });
  }

  async function handleRun() {
    if (!selectedDocId) {
      setError("Pick a document first.");
      return;
    }
    if (!structured) {
      setError("Upload the incomplete structured JSON to enrich.");
      return;
    }
    resetRun();
    setRunning(true);
    setPhase("embed", "active");

    // The embedding model, retrieval, reranking and the LLM all come from the
    // backend config. We send the uploaded incomplete JSON to be enriched.
    const body = { doc_id: selectedDocId, structured };

    try {
      await api.stream("/enrichment/run", body, (ev) => {
        switch (ev.stage) {
          case "build": {
            const be = ev.event || {};
            if (be.stage === "error") {
              setError(be.message);
              setPhase("embed", "error");
            } else if (be.stage === "done") {
              setPhase("embed", "done");
              setBuildMsg(be.message || "Embedded.");
              if (be.embedding_cost_usd) {
                setCost((c) => ({ ...c, embedding: c.embedding + be.embedding_cost_usd }));
              }
            } else if (be.stage === "chunking") {
              setBuildMsg(be.status === "done" ? `Chunked (${be.num_chunks}).` : "Chunking...");
            } else if (be.stage === "embedding") {
              setBuildMsg(be.status === "done" ? "Embedded chunks." : "Embedding chunks...");
            }
            break;
          }
          case "queries":
            setPhase("embed", "done");
            setPhase("fields", "done");
            setQueriesInfo({
              count: ev.count,
              labels: ev.labels || [],
              totalAvailable: ev.total_available ?? ev.count,
              limited: !!ev.limited,
            });
            setAttributes(
              (ev.labels || []).map((label, index) => ({ index, label, status: "pending" }))
            );
            break;
          case "generate_queries":
            if (ev.status === "start") {
              setPhase("genq", "active");
            } else if (ev.status === "progress") {
              setGenqDone(ev.completed ?? (ev.index || 0) + 1);
              if (ev.label) {
                updateAttr(ev.index, { query: ev.query });
              }
            } else if (ev.status === "done") {
              setPhase("genq", "done");
              setGenqDone(queriesInfo?.count || genqDone);
              if (ev.cost_usd) {
                setCost((c) => ({ ...c, generation: c.generation + ev.cost_usd }));
              }
            }
            break;
          case "answers":
            if (ev.status === "start") {
              setPhase("retrieve", "active");
              setPhase("rerank", "active");
              setPhase("generate", "active");
            }
            break;
          case "attribute":
            if (ev.status === "start") {
              setCurrentLabel(ev.label);
              updateAttr(ev.index, { label: ev.label, question: ev.question, status: "running" });
            } else if (ev.status === "retrieve") {
              markSubStage("retrieve");
              setActiveSub("retrieve");
              updateAttr(ev.index, { status: "retrieve" });
            } else if (ev.status === "rerank") {
              markSubStage("rerank");
              setActiveSub("rerank");
              updateAttr(ev.index, { status: "rerank" });
            } else if (ev.status === "generate") {
              markSubStage("generate");
              setActiveSub("generate");
              updateAttr(ev.index, { status: "generate" });
            } else if (ev.status === "done") {
              if (ev.cost_usd) {
                setCost((c) => ({ ...c, generation: c.generation + ev.cost_usd }));
              }
              updateAttr(ev.index, {
                status: "done",
                raw_name: ev.raw_name,
                question: ev.question,
                answer: ev.answer,
                available: ev.available,
                confidence: ev.confidence,
                normalized: ev.normalized,
                value: ev.value,
                source_pages: ev.source_pages || [],
                ground_truth: ev.ground_truth,
                original_value: ev.original_value,
                was_null: ev.was_null,
                auto_write: ev.auto_write,
                has_gt: ev.has_gt,
                matches_gt: ev.matches_gt,
                category: ev.category,
                chunks: ev.chunks || [],
                pages: ev.pages || [],
              });
            }
            break;
          case "complete":
            setPhase("retrieve", "done");
            setPhase("rerank", "done");
            setPhase("generate", "done");
            setPhase("output", "done");
            setCurrentLabel(null);
            setActiveSub(null);
            setEnrichedJson(ev.enriched_json || {});
            setSummary(ev.summary || null);
            break;
          case "error":
            setError(ev.message);
            break;
          default:
            break;
        }
      });
    } catch (e) {
      setError(`Enrichment failed: ${e.message}`);
    } finally {
      setRunning(false);
    }
  }

  const selectedAttr = attributes.find((a) => a.label === selectedLabel) || null;
  const answered = attributes.filter((a) => a.status === "done").length;

  function setVerdict(label, verdict) {
    setHumanVerdicts((prev) => {
      const next = { ...prev };
      if (next[label] === verdict) delete next[label];
      else next[label] = verdict;
      return next;
    });
  }

  function handleDownload() {
    if (!enrichedJson) return;
    // Start from the backend-enriched record (nulls filled + high-confidence
    // overwrites) then re-apply human verdicts by exact key: "correct" writes our
    // proposed value, "incorrect" reverts to the original value.
    const out = JSON.parse(JSON.stringify(enrichedJson));
    const data = out && typeof out.data === "object" && out.data ? out.data : out;
    attributes.forEach((a) => {
      if (a.status !== "done" || !a.raw_name) return;
      const verdict = humanVerdicts[a.label];
      if (!verdict) return;
      if (verdict === "correct" && a.value != null) {
        data[a.raw_name] = a.value;
      } else if (verdict === "incorrect") {
        data[a.raw_name] = a.original_value ?? null;
      }
    });
    const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${selectedDocId || "document"}_enriched.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const canRun = useMemo(
    () => !!selectedDocId && !!selectedDoc?.has_datalab && !!structured && !running,
    [selectedDocId, selectedDoc, structured, running]
  );

  return (
    <div className="app">
      <header className="topbar">
        <button type="button" className="home-back" onClick={onHome}>
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
               strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="19" y1="12" x2="5" y2="12" />
            <polyline points="12 19 5 12 12 5" />
          </svg>
          Home
        </button>
        <h1>Metadata Enrichment</h1>
        <span className="subtle">
          {"Fill null fields, validate prefilled data, review with a human in the loop"}
        </span>
      </header>

      {error && <div className="error-bar">{error}</div>}

      <div className="layout">
        <aside className="sidebar">
          <div className="panel">
            <div className="panel-title">Document</div>

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
              <select
                className="select"
                value={selectedDocId}
                onChange={(e) => {
                  const id = e.target.value;
                  setSelectedDocId(id);
                  resetRun();
                  setUploadMsg("");
                  clearStructured();
                  if (id) {
                    const doc = documents.find((d) => d.id === id);
                    if (doc?.has_matched) loadMatchedJson(id);
                  }
                }}
              >
                <option value="">-- select --</option>
                {documents.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.id}
                    {d.has_pdf ? "" : " (no pdf)"}
                  </option>
                ))}
              </select>
            </div>

            {uploadMsg && <div className="hint">{uploadMsg}</div>}

            {selectedDoc && (
              <div className="doc-status">
                <StatusDot ok={selectedDoc.has_pdf} label="PDF" />
                <StatusDot ok={selectedDoc.has_datalab} label="datalab" />
                <StatusDot ok={selectedDoc.has_matched} label="structured" />
              </div>
            )}
          </div>

          <div className="panel">
            <div className="panel-title">Structured JSON</div>
            <label className="upload-btn">
              <input
                ref={jsonRef}
                type="file"
                accept="application/json,.json"
                onChange={handleJsonUpload}
                hidden
              />
              {structured ? "Replace JSON" : "Upload incomplete JSON"}
            </label>
            {structured ? (
              <>
                <div className="doc-status" style={{ marginTop: 8 }}>
                  <StatusDot
                    ok={true}
                    label={`${structuredName || "record"} · ${structuredCount} fields${
                      structuredSource === "auto" ? " (on file)" : ""
                    }`}
                  />
                </div>
                {structuredSource === "auto" && (
                  <div className="hint subtle panel-hint">
                    A matching structured JSON was found for this document and loaded
                    automatically. Every field becomes a query. Replace it only to
                    enrich a different JSON.
                  </div>
                )}
              </>
            ) : (
              <div className="hint subtle panel-hint">
                Upload the OSDU JSON to enrich. Null fields get filled; prefilled
                fields get validated. Only changed keys differ in the output.
              </div>
            )}
            {jsonMsg && <div className="hint">{jsonMsg}</div>}
          </div>

          <div className="panel">
            <div className="panel-title">Run</div>
            <div className="hint subtle">
              The embedding model, retrieval, reranking and the LLM are all fixed by the
              backend config.
            </div>
            <button type="button" className="run-btn" onClick={handleRun} disabled={!canRun}>
              {running ? "Enriching..." : "Run enrichment"}
            </button>
            {buildMsg && <div className="hint subtle" style={{ marginTop: 8 }}>{buildMsg}</div>}
          </div>

          {(running || summary) && (
            <div className="panel">
              <div className="panel-title">Cost</div>
              <div className="cost-total">${(cost.embedding + cost.generation).toFixed(4)}</div>
              <div className="cost-rows">
                <div className="cost-row">
                  <span className="subtle">Embedding</span>
                  <span>${cost.embedding.toFixed(4)}</span>
                </div>
                <div className="cost-row">
                  <span className="subtle">Generation</span>
                  <span>${cost.generation.toFixed(4)}</span>
                </div>
              </div>
              <div className="hint subtle" style={{ marginTop: 8 }}>
                Estimated cost to enrich this document.
              </div>
            </div>
          )}

          {enrichedJson && (
            <div className="panel">
              <div className="panel-title">Output</div>
              <button type="button" className="run-btn secondary" onClick={handleDownload}>
                Download enriched JSON
              </button>
              <div className="hint subtle" style={{ marginTop: 8 }}>
                Same JSON, with only changed keys differing. Reflects your verify verdicts.
              </div>
            </div>
          )}
        </aside>

        <main className="main">
          <section className="card flow-card">
            <ProcessTimeline
              phases={phases}
              answered={answered}
              total={queriesInfo?.count || 0}
              genqDone={genqDone}
              onBlock={(id) => setDetailBlock(detailBlock === id ? null : id)}
              activeBlock={detailBlock}
            />
            <LoopProgress
              running={running}
              answered={answered}
              total={queriesInfo?.count || 0}
              currentLabel={currentLabel}
              activeSub={activeSub}
              cost={cost}
            />
            {detailBlock && (
              <div className="flow-detail">
                {(detailBlock === "genq" || detailBlock === "fields") && queriesInfo ? (
                  <div className="detail-queries">
                    <div className="subtle">
                      {queriesInfo.count} attribute queries
                      {queriesInfo.limited
                        ? ` (top ${queriesInfo.count} of ${queriesInfo.totalAvailable} fields — demo limit)`
                        : ""}
                    </div>
                    <ul>
                      {attributes.map((a) => (
                        <li key={a.label}>
                          <strong>{a.label}</strong>
                          {a.query ? <span className="subtle"> — {a.query}</span> : null}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : detailBlock === "embed" ? (
                  <div className="subtle">{buildMsg || TIMELINE[0].hint}</div>
                ) : (
                  <div className="subtle">
                    {TIMELINE.find((t) => t.id === detailBlock)?.hint} · {answered}/
                    {queriesInfo?.count || 0} attributes processed.
                  </div>
                )}
              </div>
            )}
          </section>

          {summary && (
            <section className="card">
              <div className="section-head">Dashboard</div>
              <Dashboard attributes={attributes} humanVerdicts={humanVerdicts} />
            </section>
          )}

          <div className="results">
            <div className="results-left">
              <section className="card">
                <div className="section-head">
                  Attributes <span className="count">{attributes.length}</span>
                </div>
                <table className="attr-table">
                  <thead>
                    <tr>
                      <th>Attribute</th>
                      <th>Value</th>
                      <th>Confidence</th>
                      <th>Status</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {attributes.map((a) => {
                      const verdict = humanVerdicts[a.label];
                      const cat = a.status === "done" ? effectiveCategory(a, verdict) : null;
                      const tone = cat ? toneForCategory(cat) : "muted";
                      const isSel = selectedLabel === a.label;
                      const canVerify = a.status === "done" && a.category !== "validated_correct";
                      return (
                        <tr key={a.label} className={`attr-row ${tone} ${isSel ? "sel" : ""}`}>
                          <td className="attr-name">{a.label}</td>
                          <td className="attr-value">
                            {a.status !== "done" ? (
                              <span className="subtle">
                                {a.status === "pending" ? "queued" : `${a.status}...`}
                              </span>
                            ) : a.value ? (
                              <span>{a.value}</span>
                            ) : (
                              <span className="subtle">null</span>
                            )}
                            {a.status === "done" && a.has_gt && (
                              <div className="attr-gt subtle">structured: {String(a.ground_truth)}</div>
                            )}
                          </td>
                          <td className="attr-conf">
                            {a.status === "done" ? <ConfidenceBar value={a.confidence} /> : null}
                          </td>
                          <td>
                            {cat && (
                              <span className={`pill ${tone}`}>
                                {verdict ? (verdict === "correct" ? "you: correct" : "you: incorrect")
                                  : CATEGORY_LABEL[cat] || cat}
                              </span>
                            )}
                          </td>
                          <td>
                            {canVerify && (
                              <button
                                type="button"
                                className={`hitl-btn ${isSel ? "on" : ""}`}
                                onClick={() => {
                                  setSelectedLabel(isSel ? null : a.label);
                                  setSelectedChunkId(null);
                                }}
                              >
                                Verify
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </section>

              {selectedAttr && (
                <section className="card hitl-card">
                  <div className="section-head">
                    Verify: <strong>{selectedAttr.label}</strong>
                  </div>
                  <div className="hitl-answer">
                    <div>
                      <span className="subtle">Our answer: </span>
                      {selectedAttr.value || <span className="subtle">null</span>}
                    </div>
                    {selectedAttr.has_gt && (
                      <div className="subtle">Structured value: {String(selectedAttr.ground_truth)}</div>
                    )}
                    <div className="hitl-verdict">
                      <button
                        type="button"
                        className={`verdict-btn good ${humanVerdicts[selectedAttr.label] === "correct" ? "on" : ""}`}
                        onClick={() => setVerdict(selectedAttr.label, "correct")}
                      >
                        &#10003; Correct
                      </button>
                      <button
                        type="button"
                        className={`verdict-btn bad ${humanVerdicts[selectedAttr.label] === "incorrect" ? "on" : ""}`}
                        onClick={() => setVerdict(selectedAttr.label, "incorrect")}
                      >
                        &#10007; Incorrect
                      </button>
                    </div>
                  </div>
                  <ChunksView
                    title="Retrieved chunks"
                    chunks={selectedAttr.chunks}
                    selectedId={selectedChunkId}
                    onSelect={setSelectedChunkId}
                    badge="for this attribute"
                  />
                </section>
              )}
            </div>

            <div className="results-right">
              <PdfViewer
                docId={selectedDoc?.has_pdf && selectedAttr ? selectedDocId : ""}
                highlightPages={selectedAttr?.pages || []}
                selectedChunkId={selectedChunkId}
              />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
