import React from "react";

function Badge({ label, value }) {
  let cls = "na";
  let text = "N/A";
  if (value === true) {
    cls = "pass";
    text = "PASS";
  } else if (value === false) {
    cls = "fail";
    text = "FAIL";
  }
  return (
    <span className={`vbadge ${cls}`}>
      {label}: {text}
    </span>
  );
}

export default function AnswerValidation({ answer, usage, genCost, validation }) {
  if (!answer && !validation) return null;
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
              <Badge label="Retrieval" value={validation.retrieval_hit} />
              <Badge label="Token" value={validation.token_match} />
              <Badge label="Judge" value={validation.judge_match} />
            </div>
          )}
          {!validation.skipped && validation.judge_confidence != null && (
            <div className="conf-row" title="How certain the LLM judge is of its YES/NO verdict">
              <span className="conf-label">Judge confidence</span>
              <span className="conf-bar">
                <span
                  className="conf-fill"
                  style={{ width: `${validation.judge_confidence}%` }}
                />
              </span>
              <span className="conf-val">{validation.judge_confidence}%</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
