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
  { id: "genq", label: "Queries", hint: "Reuse cached queries; generate + cache any new ones" },
  { id: "retrieve", label: "Retrieval", hint: "Find the most relevant chunks per attribute" },
  { id: "rerank", label: "Reranking", hint: "Cross-encoder re-scores the shortlist" },
  { id: "generate", label: "Answer", hint: "LLM answers each attribute" },
  { id: "output", label: "Review", hint: "Verify + correct values, then save the ground truth" },
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

export default function GroundTruth({ onHome }) {
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
  const [coreLabels, setCoreLabels] = useState([]);
  const [genqDone, setGenqDone] = useState(0);
  const [attributes, setAttributes] = useState([]);
  const [currentLabel, setCurrentLabel] = useState(null);
  const [activeSub, setActiveSub] = useState(null);
  const [cost, setCost] = useState({ embedding: 0, generation: 0 });
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");

  const [selectedLabel, setSelectedLabel] = useState(null);
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [humanVerdicts, setHumanVerdicts] = useState({});
  const [corrections, setCorrections] = useState({});
  const [detailBlock, setDetailBlock] = useState(null);
  const [saveMsg, setSaveMsg] = useState("");
  const [saving, setSaving] = useState(false);

  const verifyRef = useRef(null);

  const selectedDoc = documents.find((d) => d.id === selectedDocId);

  // A core attribute is one whose (normalized) label is a hand-written query.
  const coreSet = useMemo(() => {
    const norm = (s) => (s || "").trim().toLowerCase().replace(/[*\d]+$/, "").trim();
    return new Set(coreLabels.map(norm));
  }, [coreLabels]);

  function isCore(label) {
    const norm = (label || "").trim().toLowerCase().replace(/[*\d]+$/, "").trim();
    return coreSet.has(norm);
  }

  async function refreshDocuments() {
    const { documents } = await api.groundTruthDocuments();
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

  // When a row's Verify is clicked, bring the verification panel into view.
  useEffect(() => {
    if (selectedLabel && verifyRef.current) {
      verifyRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [selectedLabel]);

  function clearStructured() {
    setStructured(null);
    setStructuredName("");
    setStructuredCount(0);
    setStructuredSource(null);
    setJsonMsg("");
  }

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
    setCoreLabels([]);
    setGenqDone(0);
    setAttributes([]);
    setCurrentLabel(null);
    setActiveSub(null);
    setCost({ embedding: 0, generation: 0 });
    setSummary(null);
    setSelectedLabel(null);
    setSelectedChunkId(null);
    setHumanVerdicts({});
    setCorrections({});
    setDetailBlock(null);
    setSaveMsg("");
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
      setError("Upload the structured JSON to enrich.");
      return;
    }
    resetRun();
    setRunning(true);
    setPhase("embed", "active");

    const body = { doc_id: selectedDocId, structured };

    try {
      await api.stream("/ground-truth/run", body, (ev) => {
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
            setCoreLabels(ev.core_labels || []);
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
                updateAttr(ev.index, { query: ev.query, querySource: ev.source });
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

  function setCorrection(label, value) {
    setCorrections((prev) => ({ ...prev, [label]: value }));
  }

  // The page(s) the answer was taken from: prefer the LLM-cited source pages;
  // otherwise fall back to the page(s) of the top retrieved chunk.
  function answerPages(attr) {
    if (attr.source_pages && attr.source_pages.length) return attr.source_pages;
    const top = (attr.chunks || [])[0];
    if (top) {
      if (Array.isArray(top.pages) && top.pages.length) return top.pages;
      if (top.page != null) return [top.page];
    }
    return [];
  }

  // Resolve the value that should be saved for one attribute: the typed
  // correction wins; else the model answer when marked correct; else the
  // original structured value (null when it was null).
  function resolvedValue(attr) {
    const typed = corrections[attr.label];
    if (typed !== undefined && typed !== "") return typed;
    const verdict = humanVerdicts[attr.label];
    if (verdict === "correct" && attr.value != null) return attr.value;
    return attr.original_value ?? null;
  }

  // The attributes that make it into the enhanced ground truth:
  //  - every core attribute (always), plus any non-core attribute that has an
  //    established ground-truth outcome:
  //  - auto-validated (our answer matched the structured ground truth), or
  //  - the human verified it (correct -> our value; incorrect -> keep the
  //    original ground truth).
  function includeInSave(a) {
    if (a.status !== "done" || !a.raw_name) return false;
    if (isCore(a.label)) return true;
    const verdict = humanVerdicts[a.label];
    if (verdict === "correct" || verdict === "incorrect") return true;
    if (a.category === "validated_correct") return true;
    return false;
  }

  function selectedForSave() {
    return attributes.filter(includeInSave);
  }

  function buildData() {
    const data = {};
    selectedForSave().forEach((a) => {
      data[a.raw_name] = resolvedValue(a);
    });
    return data;
  }

  async function handleSave() {
    const data = buildData();
    if (Object.keys(data).length === 0) {
      setSaveMsg("Nothing to save yet — run the pipeline first.");
      return;
    }
    setSaving(true);
    setSaveMsg("");
    try {
      const res = await api.groundTruthSave({ doc_id: selectedDocId, data });
      setSaveMsg(`Saved ${res.count} attributes -> ${res.path}`);
    } catch (e) {
      setSaveMsg(`Save failed: ${e.message}`);
    } finally {
      setSaving(false);
    }
    // Also offer a browser download of the same {"data": {...}} shape.
    const blob = new Blob([JSON.stringify({ data }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${selectedDocId || "document"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const canRun = useMemo(
    () => !!selectedDocId && !!selectedDoc?.has_datalab && !!structured && !running,
    [selectedDocId, selectedDoc, structured, running]
  );

  const savedCount = selectedForSave().length;

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
        <h1>Ground Truth Enrichment</h1>
        <span className="subtle">
          {"Verify + correct core attributes, then save an enhanced ground-truth dataset"}
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
              {structured ? "Replace JSON" : "Upload structured JSON"}
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
                    automatically. Every field is queried so you can verify + correct it.
                  </div>
                )}
              </>
            ) : (
              <div className="hint subtle panel-hint">
                Upload the OSDU JSON. The pipeline answers every attribute so you can
                fix wrong / null ground truth before saving.
              </div>
            )}
            {jsonMsg && <div className="hint">{jsonMsg}</div>}
          </div>

          <div className="panel">
            <div className="panel-title">Run</div>
            <div className="hint subtle">
              Queries are reused from queries.txt when available and generated (then
              cached) otherwise. Models come from the backend config.
            </div>
            <button type="button" className="run-btn" onClick={handleRun} disabled={!canRun}>
              {running ? "Running..." : "Run enrichment"}
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
            </div>
          )}

          {summary && (
            <div className="panel">
              <div className="panel-title">Save ground truth</div>
              <div className="hint subtle" style={{ marginBottom: 8 }}>
                {savedCount} attributes will be saved (all core attributes, plus
                any non-core field that was auto-validated or that you verified —
                correct keeps our value, incorrect keeps the original ground truth).
              </div>
              <button
                type="button"
                className="run-btn secondary"
                onClick={handleSave}
                disabled={saving || savedCount === 0}
              >
                {saving ? "Saving..." : "Save enhanced ground truth"}
              </button>
              {saveMsg && <div className="hint" style={{ marginTop: 8 }}>{saveMsg}</div>}
              <div className="hint subtle" style={{ marginTop: 8 }}>
                Written to enriched_ground_truth/{selectedDocId || "&lt;doc&gt;"}.json and
                downloaded to your browser.
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
                    </div>
                    <ul>
                      {attributes.map((a) => (
                        <li key={a.label}>
                          <strong>{a.label}</strong>
                          {a.querySource && (
                            <span className={`pill ${a.querySource === "cache" ? "info" : "enriched"}`} style={{ marginLeft: 6 }}>
                              {a.querySource}
                            </span>
                          )}
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
                      const core = isCore(a.label);
                      return (
                        <tr key={a.label} className={`attr-row ${tone} ${isSel ? "sel" : ""}`}>
                          <td className="attr-name">
                            {a.label}
                            {core && <span className="pill info gt-core-pill">core</span>}
                          </td>
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
                            {a.status === "done" && a.available && answerPages(a).length > 0 && (
                              <div className="attr-src subtle">
                                source: p. {answerPages(a).join(", ")}
                              </div>
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
                            {a.status === "done" && (
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
                <section className="card hitl-card" ref={verifyRef}>
                  <div className="section-head">
                    Verify: <strong>{selectedAttr.label}</strong>
                    {isCore(selectedAttr.label) && <span className="pill info gt-core-pill">core</span>}
                  </div>
                  <div className="hitl-answer">
                    <div>
                      <span className="subtle">Our answer: </span>
                      {selectedAttr.value || <span className="subtle">null</span>}
                    </div>
                    {selectedAttr.has_gt && (
                      <div className="subtle">Structured value: {String(selectedAttr.ground_truth)}</div>
                    )}
                    {answerPages(selectedAttr).length > 0 && (
                      <div className="subtle">
                        Source page(s): {answerPages(selectedAttr).join(", ")}
                      </div>
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
                    <div className="field gt-correction">
                      <label>Corrected value (overrides what gets saved)</label>
                      <input
                        type="text"
                        className="gt-correction-input"
                        placeholder={
                          selectedAttr.value
                            ? `default: ${selectedAttr.value}`
                            : "type the correct value"
                        }
                        value={corrections[selectedAttr.label] ?? ""}
                        onChange={(e) => setCorrection(selectedAttr.label, e.target.value)}
                      />
                      <div className="hint subtle">
                        Will be saved as: <strong>{String(resolvedValue(selectedAttr))}</strong>
                      </div>
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
