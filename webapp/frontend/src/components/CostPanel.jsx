import React from "react";

function fmtCost(v) {
  return v == null ? "-" : `$${Number(v).toFixed(4)}`;
}

function fmtTokens(usage) {
  if (!usage) return "-";
  const i = usage.input_tokens ?? 0;
  const o = usage.output_tokens ?? 0;
  if (!i && !o) return "-";
  return `${i} in / ${o} out`;
}

export default function CostPanel({ embedCost, genUsage, genCost, verifyUsage, verifyCost }) {
  const hasAny =
    embedCost != null || genCost != null || verifyCost != null || genUsage || verifyUsage;
  if (!hasAny) return null;

  const total = (embedCost || 0) + (genCost || 0) + (verifyCost || 0);

  return (
    <div className="cost-panel">
      <div className="cost-title">Cost &amp; Tokens</div>
      <table className="cost-table">
        <thead>
          <tr>
            <th>Stage</th>
            <th>Tokens</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Embedding</td>
            <td>-</td>
            <td>{fmtCost(embedCost)}</td>
          </tr>
          <tr>
            <td>Generation</td>
            <td>{fmtTokens(genUsage)}</td>
            <td>{fmtCost(genCost)}</td>
          </tr>
          <tr>
            <td>Verification</td>
            <td>{fmtTokens(verifyUsage)}</td>
            <td>{fmtCost(verifyCost)}</td>
          </tr>
          <tr className="cost-total">
            <td>Total</td>
            <td />
            <td>{fmtCost(total)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
