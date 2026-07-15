import React from "react";

const VERDICT_META = {
  consistent: { cls: "pass", text: "CONSISTENT" },
  inconsistent: { cls: "fail", text: "INCONSISTENT" },
  not_found: { cls: "na", text: "NOT FOUND" },
};

function CheckItem({ c, running, isCurrent }) {
  let badge;
  if (c.verdict) {
    const meta = VERDICT_META[c.verdict] || VERDICT_META.not_found;
    badge = <span className={`vbadge ${meta.cls}`}>{meta.text}</span>;
  } else if (running && isCurrent) {
    badge = (
      <span className="vbadge na conf-check-running">
        <span className="spin" /> checking...
      </span>
    );
  } else {
    badge = <span className="vbadge na">pending</span>;
  }
  return (
    <li className={`conf-check ${running && isCurrent ? "current" : ""}`}>
      <div className="conf-check-head">
        {badge}
        <span className="conf-check-rel">{c.relevance}</span>
      </div>
      <div className="conf-check-q">{c.question}</div>
      {c.rule && <div className="conf-check-rule">rule: {c.rule}</div>}
      {c.related_fact && <div className="conf-check-fact">found: {c.related_fact}</div>}
    </li>
  );
}

function SuggestionBox({ suggestion, reverifying, onAccept, onReject }) {
  if (!suggestion || !suggestion.suggested_answer) return null;
  return (
    <div className="suggestion-box">
      <div className="suggestion-title">Low confidence - suggested correction</div>
      <div className="suggestion-value">{suggestion.suggested_answer}</div>
      {suggestion.reason && <div className="suggestion-reason">{suggestion.reason}</div>}
      <div className="suggestion-actions">
        <button
          className="btn primary sm"
          disabled={reverifying}
          onClick={() => onAccept && onAccept(suggestion.suggested_answer)}
        >
          {reverifying ? "Re-verifying..." : "Accept & re-verify"}
        </button>
        <button className="btn sm" disabled={reverifying} onClick={() => onReject && onReject()}>
          Reject
        </button>
      </div>
    </div>
  );
}

function ConfidenceSection({ confidence, reverifying, onAcceptSuggestion, onRejectSuggestion }) {
  if (!confidence) return null;
  const {
    running, planning, confidence: score, band, checks = [],
    breakdown, verbalized_confidence, total, currentIndex, suggestion,
  } = confidence;

  const doneCount = checks.filter((c) => c && c.verdict).length;
  const totalCount = total ?? checks.length;

  return (
    <div className="validation">
      <div className="validation-title">
        Confidence (CoVe verification)
        {running ? (
          <span className="badge subtle">
            {planning ? "planning checks..." : `verifying ${doneCount}/${totalCount}`}
          </span>
        ) : (
          <span className={`badge conf-band-${band || "low"}`}>{band || "-"}</span>
        )}
      </div>

      {!running && score != null && (
        <div className="conf-row" title="Confidence from indirect corroboration of the answer">
          <span className="conf-label">Confidence</span>
          <span className="conf-bar">
            <span className="conf-fill" style={{ width: `${score}%` }} />
          </span>
          <span className="conf-val">{score}%</span>
        </div>
      )}

      {!running && verbalized_confidence != null && (
        <div className="answer-usage">
          model self-reported: {verbalized_confidence}%
          {breakdown?.vetoed && <> &middot; contradiction veto applied</>}
        </div>
      )}

      {!running && suggestion && (
        <SuggestionBox
          suggestion={suggestion}
          reverifying={reverifying}
          onAccept={onAcceptSuggestion}
          onReject={onRejectSuggestion}
        />
      )}

      {running && planning && (
        <div className="answer-usage">Generating verification checks...</div>
      )}

      {checks.length > 0 && (
        running ? (
          <ul className="conf-check-list">
            {checks.map((c, i) => (
              <CheckItem key={i} c={c} running={running} isCurrent={i === currentIndex} />
            ))}
          </ul>
        ) : (
          <details className="conf-checks">
            <summary>{checks.length} verification checks</summary>
            <ul className="conf-check-list">
              {checks.map((c, i) => (
                <CheckItem key={i} c={c} running={false} isCurrent={false} />
              ))}
            </ul>
          </details>
        )
      )}
    </div>
  );
}

export default function AnswerValidation({
  answer, usage, genCost, confidence, validation,
  reverifying, onAcceptSuggestion, onRejectSuggestion,
}) {
  if (!answer && !validation && !confidence) return null;
  return (
    <div className="answer-card">
      {answer && (
        <>
          <div className="answer-title">LLM Answer</div>
          <div className="answer-text">{answer}</div>
          {usage && (
            <div className="answer-usage">
              tokens: {usage.input_tokens} in / {usage.output_tokens} out
              {genCost != null && <> &middot; cost: ${genCost.toFixed(4)}</>}
            </div>
          )}
        </>
      )}
      {confidence && (
        <ConfidenceSection
          confidence={confidence}
          reverifying={reverifying}
          onAcceptSuggestion={onAcceptSuggestion}
          onRejectSuggestion={onRejectSuggestion}
        />
      )}
      {validation && (
        <div className="validation">
          <div className="validation-title">
            Validation
            {validation.skipped ? (
              <span className="badge subtle">no ground truth</span>
            ) : (
              <span className="gt">expected: {String(validation.ground_truth)}</span>
            )}
          </div>
          {!validation.skipped && (
            <div className="vbadges">
              <span className={`vbadge ${validation.judge_match ? "pass" : "fail"}`}>
                {validation.judge_match ? "VALIDATED" : "NOT VALIDATED"}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
