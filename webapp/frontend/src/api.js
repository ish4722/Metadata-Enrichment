// REST + NDJSON streaming client for the RAG Explorer backend.

const BASE = "/api";

async function getJson(path) {
  const resp = await fetch(`${BASE}${path}`);
  if (!resp.ok) throw new Error(`${path} -> ${resp.status}`);
  return resp.json();
}

export const api = {
  health: () => getJson("/health"),
  documents: () => getJson("/documents"),
  document: (id) => getJson(`/documents/${id}`),
  embeddingModels: () => getJson("/embedding-models"),
  queries: () => getJson("/queries"),
  enrichmentDocuments: () => getJson("/enrichment/documents"),
  groundTruthDocuments: () => getJson("/ground-truth/documents"),
  matched: (id) => getJson(`/matched/${id}`),
  pdfUrl: (id) => `${BASE}/pdf/${id}`,

  async groundTruthSave(body) {
    const resp = await fetch(`${BASE}/ground-truth/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`ground-truth/save -> ${resp.status} ${text}`);
    }
    return resp.json();
  },

  async upload(file) {
    const form = new FormData();
    form.append("file", file);
    const resp = await fetch(`${BASE}/upload`, { method: "POST", body: form });
    if (!resp.ok) throw new Error(`upload -> ${resp.status}`);
    return resp.json();
  },

  // POST `body` and invoke `onEvent` for each NDJSON line as it streams in.
  async stream(path, body, onEvent, signal) {
    const resp = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!resp.ok || !resp.body) {
      const text = await resp.text().catch(() => "");
      throw new Error(`${path} -> ${resp.status} ${text}`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const flushLine = (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      try {
        onEvent(JSON.parse(trimmed));
      } catch (e) {
        // ignore partial / malformed lines
      }
    };
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        flushLine(buffer.slice(0, idx));
        buffer = buffer.slice(idx + 1);
      }
    }
    flushLine(buffer);
  },
};
