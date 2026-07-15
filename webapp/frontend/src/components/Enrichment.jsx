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
  { id: "verify", label: "Verify (CoVe)", hint: "Chain-of-Verification confidence per answer" },
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
  { id: "verify", label: "Verify (CoVe)" },
];

function LoopProgress({ running, answered, total, currentLabel, activeSub }) {
  const pctDone = total ? Math.round((answered / total) * 100) : 0;
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
      </div>
    </div>
  );
}

// ── categories / verdicts ─────────────────────────────────────────────────────
const CATEGORY_LABEL = {
  validated_correct: "validated",
  validated_incorrect: "mismatch",
  updated: "updated",
  enriched_confident: "enriched (high)",
  enriched_medium: "enriched (medium)",
  enriched_low_conf: "low confidence",
  not_found: "not found",
};

// CoVe confidence -> band (matches the backend BAND_HIGH=0.75 / BAND_MEDIUM=0.5).
function bandForConfidence(v) {
  if (v === null || v === undefined) return null;
  if (v >= 0.75) return "high";
  if (v >= 0.5) return "medium";
  return "low";
}

// A human edit promotes the answer to high confidence (requirement 6).
function effectiveBand(attr, edit) {
  if (edit) return "high";
  return attr.band || bandForConfidence(attr.confidence);
}

function effectiveValue(attr, edit) {
  return edit && edit.value != null && edit.value !== "" ? edit.value : attr.value;
}

function effectiveCategory(attr, verdict, band) {
  if (verdict === "correct") return "validated_correct";
  if (verdict === "incorrect") return "validated_incorrect";
  if (attr.has_gt) {
    if (attr.matches_gt) return "validated_correct";
    // A high-confidence disagreement is a confident UPDATE (blue), not a
    // red mismatch: the model overwrote the ground truth on purpose.
    return band === "high" ? "updated" : "validated_incorrect";
  }
  if (!attr.available || attr.value == null) return "not_found";
  if (band === "high") return "enriched_confident";
  if (band === "medium") return "enriched_medium";
  return "enriched_low_conf";
}

function toneForCategory(cat) {
  switch (cat) {
    case "validated_correct":
      return "good";
    case "validated_incorrect":
      return "bad";
    case "updated":
      return "enriched";
    case "enriched_confident":
      return "enriched";
    case "enriched_medium":
      return "warn";
    case "enriched_low_conf":
      return "warn";
    default:
      return "muted";
  }
}

// ── dashboard ─────────────────────────────────────────────────────────────────
function Dashboard({ attributes, humanVerdicts, humanEdits }) {
  const done = attributes.filter((a) => a.status === "done");

  let validatedCorrect = 0;
  let withGt = 0;
  let highUpdate = 0;
  let mediumFill = 0;
  let enriched = 0;

  done.forEach((a) => {
    const edit = humanEdits[a.label];
    const verdict = humanVerdicts[a.label];
    const band = effectiveBand(a, edit);
    const value = effectiveValue(a, edit);
    const hasAnswer = a.available && value != null && String(value).trim() !== "";
    // Use the SAME category the row status pill uses so the dashboard totals
    // always match what the table shows. The four buckets are mutually
    // exclusive: an "enriched" (GT-null) answer is never counted as "updated".
    const cat = effectiveCategory(a, verdict, band);

    if (a.has_gt) withGt += 1;

    if (cat === "validated_correct") {
      // a) correctly validated against the ground truth.
      validatedCorrect += 1;
    } else if (cat === "updated") {
      // b) had GT, model overwrote it with HIGH confidence.
      highUpdate += 1;
    } else if (cat === "enriched_confident") {
      // d) GT was null, model produced a HIGH-confidence answer.
      enriched += 1;
    } else if (cat === "enriched_medium") {
      // c) GT null, MEDIUM-confidence fill (needs review).
      mediumFill += 1;
    } else if (cat === "validated_incorrect" && band === "medium" && hasAnswer) {
      // c) disagreed with GT at MEDIUM confidence (needs review).
      mediumFill += 1;
    }
  });

  const cards = [
    { key: "vc", label: "Validated with ground truth", value: `${validatedCorrect}`, sub: `of ${withGt} with GT`, tone: "good" },
    { key: "hu", label: "Updated (high confidence)", value: `${highUpdate}`, sub: "model filled / updated", tone: "enriched" },
    { key: "mf", label: "Filled (medium confidence)", value: `${mediumFill}`, sub: "null filled / updated earlier", tone: "warn" },
    { key: "en", label: "Enriched (GT null)", value: `${enriched}`, sub: "answer beyond ground truth", tone: "info" },
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

function ConfidenceBar({ value, band }) {
  if (value === null || value === undefined) return <span className="subtle">n/a</span>;
  const p = Math.round(value * 100);
  const b = band || bandForConfidence(value);
  const tone = b === "high" ? "good" : b === "medium" ? "warn" : "bad";
  return (
    <div className="conf-bar" title={`CoVe confidence ${p}% (${b})`}>
      <div className={`conf-fill ${tone}`} style={{ width: `${p}%` }} />
      <span className="conf-num">{p}%</span>
    </div>
  );
}

const VERDICT_META = {
  consistent: { cls: "pass", text: "CONSISTENT" },
  inconsistent: { cls: "fail", text: "INCONSISTENT" },
  not_found: { cls: "na", text: "NOT FOUND" },
};

// CoVe corroboration checks for a single attribute answer. Renders pending /
// checking / judged states so the verification can be watched live.
function CoveChecks({ checks, open }) {
  if (!checks || checks.length === 0) return null;
  const judged = checks.filter((c) => c && c.verdict).length;
  return (
    <details className="conf-checks" open={open}>
      <summary>CoVe questions ({judged}/{checks.length} judged)</summary>
      <ul className="conf-check-list">
        {checks.map((c, i) => {
          let badge;
          if (c.verdict) {
            const meta = VERDICT_META[c.verdict] || VERDICT_META.not_found;
            badge = <span className={`vbadge ${meta.cls}`}>{meta.text}</span>;
          } else if (c.running) {
            badge = (
              <span className="vbadge na">
                <span className="spin" /> checking
              </span>
            );
          } else {
            badge = <span className="vbadge na">pending</span>;
          }
          return (
            <li key={i} className={`conf-check ${c.running ? "current" : ""}`}>
              <div className="conf-check-head">
                {badge}
                <span className="conf-check-rel">{c.relevance}</span>
              </div>
              <div className="conf-check-q">{c.question}</div>
              {c.rule && <div className="conf-check-rule">rule: {c.rule}</div>}
              {c.related_fact && <div className="conf-check-fact">found: {c.related_fact}</div>}
            </li>
          );
        })}
      </ul>
    </details>
  );
}

// Full, step-by-step visualisation of one query's pipeline:
// retrieval -> reranking -> LLM answer -> CoVe verification.
function PipelineSteps({ attr, selectedChunkId, onSelect }) {
  if (!attr) return null;
  const band = attr.band;
  const confPct = attr.confidence != null ? Math.round(attr.confidence * 100) : null;
  const verbPct =
    attr.verbalized_confidence != null ? Math.round(attr.verbalized_confidence * 100) : null;
  const reranked = attr.reranked_chunks || attr.chunks || [];
  const retrieved = attr.retrieved_chunks || [];
  const bandTone = band === "high" ? "good" : band === "medium" ? "warn" : "bad";
  const sug = attr.cove_suggestion;
  return (
    <div className="pipe-steps">
      <div className="pipe-step">
        <div className="pipe-step-title">
          <span className="pipe-step-num">1</span> Retrieval
          <span className="count">{retrieved.length}</span>
        </div>
        <div className="pipe-step-body">
          <div className="subtle pipe-q">query: {attr.retrieval_query || attr.question || "—"}</div>
          <details className="pipe-chunks">
            <summary>{retrieved.length} retrieved chunks</summary>
            <ChunksView
              title=""
              chunks={retrieved}
              selectedId={selectedChunkId}
              onSelect={onSelect}
              badge="dense/hybrid"
            />
          </details>
        </div>
      </div>

      <div className="pipe-step">
        <div className="pipe-step-title">
          <span className="pipe-step-num">2</span> Reranking
          <span className="count">{reranked.length}</span>
        </div>
        <div className="pipe-step-body">
          <details className="pipe-chunks" open>
            <summary>{reranked.length} reranked chunks (top)</summary>
            <ChunksView
              title=""
              chunks={reranked}
              selectedId={selectedChunkId}
              onSelect={onSelect}
              badge="cross-encoder"
            />
          </details>
        </div>
      </div>

      <div className="pipe-step">
        <div className="pipe-step-title">
          <span className="pipe-step-num">3</span> LLM answer
        </div>
        <div className="pipe-step-body">
          <div className="pipe-answer">
            {attr.answer || <span className="subtle">no answer</span>}
          </div>
          {verbPct != null && (
            <div className="subtle">model self-reported confidence: {verbPct}%</div>
          )}
        </div>
      </div>

      <div className="pipe-step">
        <div className="pipe-step-title">
          <span className="pipe-step-num">4</span> CoVe verification
          {band && (
            <span className={`pill ${bandTone}`}>
              {band}
              {confPct != null ? ` · ${confPct}%` : ""}
            </span>
          )}
          {attr.cove_rounds > 0 && (
            <span className="pill info">
              re-verified ×{attr.cove_rounds}
            </span>
          )}
        </div>
        <div className="pipe-step-body">
          {attr.cove_rounds > 0 && (
            <div className="subtle">
              confidence was low; accepted the verifier's correction and
              re-scored {attr.cove_rounds} time{attr.cove_rounds > 1 ? "s" : ""}
              {" "}— confidence below reflects the final value.
            </div>
          )}
          {attr.cove_checks && attr.cove_checks.length > 0 ? (
            <CoveChecks checks={attr.cove_checks} open />
          ) : (
            <div className="subtle">no verification checks</div>
          )}
          {sug && sug.suggested_answer && (
            <div className="suggestion-box">
              <div className="suggestion-title">
                CoVe suggested correction{attr.auto_corrected ? " (auto-accepted)" : ""}
              </div>
              <div className="suggestion-value">{sug.suggested_answer}</div>
              {sug.reason && <div className="suggestion-reason">{sug.reason}</div>}
            </div>
          )}
        </div>
      </div>
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
  verify: "pending",
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
  // Human edits for medium-confidence answers (requirement 6): label -> new value.
  // An edited answer is promoted to high confidence.
  const [humanEdits, setHumanEdits] = useState({});
  const [editDraft, setEditDraft] = useState("");
  const [detailBlock, setDetailBlock] = useState(null);

  // Total attribute-query count captured off the stream so the "generate queries"
  // counter is not reset by a stale closure value (requirement 4).
  const genqTotalRef = useRef(0);

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

  // Pull the record to enrich from new_checking_truth/<id>.json (requirement 1),
  // falling back to matched_json, so the user need not upload one manually.
  async function loadMatchedJson(docId) {
    let record = null;
    let origin = "";
    try {
      record = await api.enrichmentSource(docId);
      origin = "new_checking_truth";
    } catch (err) {
      try {
        record = await api.matched(docId);
        origin = "matched_json";
      } catch (err2) {
        return false;
      }
    }
    const data =
      record && typeof record.data === "object" && record.data ? record.data : record;
    const count = data && typeof data === "object" ? Object.keys(data).length : 0;
    setStructured(record);
    setStructuredName(`${docId}.json`);
    setStructuredCount(count);
    setStructuredSource("auto");
    setJsonMsg(`Using the ${origin} record for ${docId} (${count} fields).`);
    return true;
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
    setHumanEdits({});
    setDetailBlock(null);
    setError("");
    genqTotalRef.current = 0;
  }

  function markSubStage(sub) {
    setPhases((prev) => {
      const order = ["retrieve", "rerank", "generate", "verify"];
      const cur = order.indexOf(sub);
      if (cur < 0) return prev;
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
            genqTotalRef.current = ev.count || 0;
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
              if (ev.count) genqTotalRef.current = ev.count;
            } else if (ev.status === "progress") {
              if (ev.total) genqTotalRef.current = ev.total;
              // Never move the counter backwards (progress events can arrive out
              // of order when queries are generated across a thread pool).
              setGenqDone((prev) => Math.max(prev, ev.completed ?? (ev.index || 0) + 1));
              if (ev.label) {
                updateAttr(ev.index, { query: ev.query });
              }
            } else if (ev.status === "done") {
              setPhase("genq", "done");
              // Show the full count (ref survives the stale-closure trap that
              // previously reset this to 0/5).
              setGenqDone(genqTotalRef.current);
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
              setPhase("verify", "active");
            }
            break;
          case "attribute":
            if (ev.status === "start") {
              setCurrentLabel(ev.label);
              updateAttr(ev.index, { label: ev.label, question: ev.question, status: "queued" });
            } else if (ev.status === "retrieve") {
              markSubStage("retrieve");
              setActiveSub("retrieve");
              setCurrentLabel(ev.label);
              updateAttr(ev.index, { status: "retrieve", retrieval_query: ev.query });
            } else if (ev.status === "retrieved") {
              updateAttr(ev.index, { retrieved_chunks: ev.chunks || [] });
            } else if (ev.status === "rerank") {
              markSubStage("rerank");
              setActiveSub("rerank");
              setCurrentLabel(ev.label);
              updateAttr(ev.index, { status: "rerank" });
            } else if (ev.status === "reranked") {
              updateAttr(ev.index, { reranked_chunks: ev.chunks || [], chunks: ev.chunks || [] });
            } else if (ev.status === "generate") {
              markSubStage("generate");
              setActiveSub("generate");
              setCurrentLabel(ev.label);
              updateAttr(ev.index, { status: "generate" });
            } else if (ev.status === "answered") {
              updateAttr(ev.index, {
                answer: ev.answer,
                available: ev.available,
                verbalized_confidence: ev.verbalized_confidence,
              });
            } else if (ev.status === "verify") {
              markSubStage("verify");
              setActiveSub("verify");
              setCurrentLabel(ev.label);
              updateAttr(ev.index, { status: "verify" });
            } else if (ev.status === "cove_plan") {
              setActiveSub("verify");
              updateAttr(ev.index, {
                cove_checks: (ev.checks || []).map((c) => ({ ...c, verdict: null })),
                ...(ev.round ? { cove_rounds: ev.round } : {}),
              });
            } else if (ev.status === "cove_retry") {
              setActiveSub("verify");
              updateAttr(ev.index, {
                cove_rounds: ev.round,
                cove_retry_answer: ev.answer,
                answer: ev.answer,
              });
            } else if (ev.status === "cove_check") {
              setActiveSub("verify");
              // Mark which check is currently being judged.
              setAttributes((prev) => {
                const next = [...prev];
                const a = next[ev.index];
                if (a) {
                  const checks = [...(a.cove_checks || [])];
                  if (checks[ev.i]) checks[ev.i] = { ...checks[ev.i], running: true };
                  next[ev.index] = { ...a, cove_checks: checks, cove_current: ev.i };
                }
                return next;
              });
            } else if (ev.status === "cove_check_done") {
              setAttributes((prev) => {
                const next = [...prev];
                const a = next[ev.index];
                if (a) {
                  const checks = [...(a.cove_checks || [])];
                  checks[ev.i] = { ...ev.check, running: false };
                  next[ev.index] = { ...a, cove_checks: checks };
                }
                return next;
              });
            } else if (ev.status === "done") {
              if (ev.cost_usd) {
                setCost((c) => ({ ...c, generation: c.generation + ev.cost_usd }));
              }
              updateAttr(ev.index, {
                status: "done",
                raw_name: ev.raw_name,
                question: ev.question,
                retrieval_query: ev.retrieval_query,
                answer: ev.answer,
                available: ev.available,
                confidence: ev.confidence,
                verbalized_confidence: ev.verbalized_confidence,
                band: ev.band,
                cove_checks: ev.cove_checks || [],
                cove_suggestion: ev.cove_suggestion || null,
                cove_rounds: ev.cove_rounds || 0,
                auto_corrected: !!ev.auto_corrected,
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
                retrieved_chunks: ev.retrieved_chunks || [],
                reranked_chunks: ev.reranked_chunks || [],
                pages: ev.pages || [],
              });
            }
            break;
          case "complete":
            setPhase("retrieve", "done");
            setPhase("rerank", "done");
            setPhase("generate", "done");
            setPhase("verify", "done");
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

  // Requirement 6: the reviewer edits a medium-confidence answer themselves.
  // Saving the edit promotes the attribute to high confidence.
  function saveEdit(label) {
    const v = (editDraft || "").trim();
    if (!v) return;
    setHumanEdits((prev) => ({ ...prev, [label]: { value: v } }));
  }

  function clearEdit(label) {
    setHumanEdits((prev) => {
      const next = { ...prev };
      delete next[label];
      return next;
    });
  }

  // Seed the edit box with the current (or already-edited) value when a new
  // attribute is opened for review.
  useEffect(() => {
    if (!selectedLabel) return;
    const attr = attributes.find((a) => a.label === selectedLabel);
    if (!attr) return;
    const edit = humanEdits[selectedLabel];
    setEditDraft(String((edit && edit.value != null ? edit.value : attr.value) ?? ""));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedLabel]);

  function handleDownload() {
    if (!enrichedJson) return;
    // Start from the backend-enriched record (nulls filled + high-confidence
    // overwrites) then re-apply human verdicts by exact key: "correct" writes our
    // proposed value, "incorrect" reverts to the original value.
    const out = JSON.parse(JSON.stringify(enrichedJson));
    const data = out && typeof out.data === "object" && out.data ? out.data : out;
    attributes.forEach((a) => {
      if (a.status !== "done" || !a.raw_name) return;
      // A human edit (requirement 6) takes precedence and is written verbatim.
      const edit = humanEdits[a.label];
      if (edit && edit.value != null && edit.value !== "") {
        data[a.raw_name] = edit.value;
        return;
      }
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
              <Dashboard attributes={attributes} humanVerdicts={humanVerdicts} humanEdits={humanEdits} />
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
                      const edit = humanEdits[a.label];
                      const band = a.status === "done" ? effectiveBand(a, edit) : null;
                      const value = a.status === "done" ? effectiveValue(a, edit) : null;
                      const cat = a.status === "done" ? effectiveCategory(a, verdict, band) : null;
                      const tone = cat ? toneForCategory(cat) : "muted";
                      const isSel = selectedLabel === a.label;
                      const canOpen = a.status !== "pending";
                      return (
                        <tr key={a.label} className={`attr-row ${tone} ${isSel ? "sel" : ""}`}>
                          <td className="attr-name">{a.label}</td>
                          <td className="attr-value">
                            {a.status !== "done" ? (
                              <span className="subtle">
                                {a.status === "pending" ? "queued" : `${a.status}...`}
                              </span>
                            ) : value ? (
                              <span>{value}</span>
                            ) : (
                              <span className="subtle">null</span>
                            )}
                            {edit && <span className="tag-edited"> edited</span>}
                            {!edit && a.auto_corrected && (
                              <span className="tag-auto" title="Auto-accepted CoVe correction (low confidence)">
                                {" "}auto
                              </span>
                            )}
                            {a.status === "done" && a.has_gt && (
                              <div className="attr-gt subtle">ground truth: {String(a.ground_truth)}</div>
                            )}
                          </td>
                          <td className="attr-conf">
                            {a.status === "done" ? (
                              <ConfidenceBar value={a.confidence} band={band} />
                            ) : null}
                          </td>
                          <td>
                            {cat && (
                              <span className={`pill ${tone}`}>
                                {verdict
                                  ? verdict === "correct"
                                    ? "you: correct"
                                    : "you: incorrect"
                                  : edit
                                  ? "edited (high)"
                                  : CATEGORY_LABEL[cat] || cat}
                              </span>
                            )}
                          </td>
                          <td>
                            {canOpen && (
                              <button
                                type="button"
                                className={`hitl-btn ${isSel ? "on" : ""}`}
                                onClick={() => {
                                  setSelectedLabel(isSel ? null : a.label);
                                  setSelectedChunkId(null);
                                }}
                              >
                                {a.status === "done" ? "Details" : "Watch"}
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </section>

              {selectedAttr && (() => {
                const edit = humanEdits[selectedAttr.label];
                const band = effectiveBand(selectedAttr, edit);
                const value = effectiveValue(selectedAttr, edit);
                const isMedium = band === "medium";
                const isDone = selectedAttr.status === "done";
                return (
                <section className="card hitl-card">
                  <div className="section-head">
                    Query steps: <strong>{selectedAttr.label}</strong>
                    {isDone ? (
                      <span className={`pill ${toneForCategory(effectiveCategory(selectedAttr, humanVerdicts[selectedAttr.label], band))}`}>
                        {edit ? "edited (high)" : `${band || "?"} confidence`}
                      </span>
                    ) : (
                      <span className="pill muted">
                        <span className="spin" /> {selectedAttr.status || "running"}…
                      </span>
                    )}
                  </div>

                  {isDone && (
                    <div className="hitl-answer">
                      <div>
                        <span className="subtle">Model answer: </span>
                        {value || <span className="subtle">null</span>}
                      </div>
                      {selectedAttr.auto_corrected && !edit && (
                        <div className="subtle">
                          Auto-accepted CoVe correction (answer had low confidence).
                        </div>
                      )}
                      {selectedAttr.has_gt && (
                        <div className="subtle">Ground truth: {String(selectedAttr.ground_truth)}</div>
                      )}

                      {/* Requirement 6: reviewer edits a medium-confidence answer;
                          saving it promotes the attribute to high confidence. */}
                      {isMedium && (
                        <div className="hitl-edit">
                          <label className="subtle">Edit the answer (use the chunks below):</label>
                          <div className="hitl-edit-row">
                            <input
                              type="text"
                              className="hitl-edit-input"
                              value={editDraft}
                              onChange={(e) => setEditDraft(e.target.value)}
                              placeholder="Corrected value"
                            />
                            <button
                              type="button"
                              className="verdict-btn good"
                              onClick={() => saveEdit(selectedAttr.label)}
                            >
                              Save &amp; mark high confidence
                            </button>
                          </div>
                        </div>
                      )}
                      {edit && (
                        <div className="hitl-edit-row" style={{ marginTop: 6 }}>
                          <span className="subtle">Your edit is applied.</span>
                          <button
                            type="button"
                            className="verdict-btn"
                            onClick={() => clearEdit(selectedAttr.label)}
                          >
                            Clear edit
                          </button>
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
                    </div>
                  )}

                  <PipelineSteps
                    attr={selectedAttr}
                    selectedChunkId={selectedChunkId}
                    onSelect={setSelectedChunkId}
                  />
                </section>
                );
              })()}
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
