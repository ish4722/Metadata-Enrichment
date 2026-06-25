import React, { useEffect, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { api } from "../api.js";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

const RENDER_WIDTH = 720;

export default function PdfViewer({ docId, highlightPages, selectedChunkId }) {
  const docRef = useRef(null);
  const containerRef = useRef(null);
  const pageRefs = useRef({}); // datalabPage -> wrapper element
  const [status, setStatus] = useState("");
  const [renderedPages, setRenderedPages] = useState([]);
  const [docToken, setDocToken] = useState(0); // bumps when a PDF finishes loading

  // Load the PDF document when the doc changes.
  useEffect(() => {
    let cancelled = false;
    setRenderedPages([]);
    docRef.current = null;
    if (!docId) return;
    setStatus("Loading PDF...");
    const task = pdfjsLib.getDocument({ url: api.pdfUrl(docId) });
    task.promise
      .then((pdf) => {
        if (cancelled) return;
        docRef.current = pdf;
        setStatus("");
        setDocToken((t) => t + 1); // signal the render effect that the doc is ready
      })
      .catch((err) => {
        if (!cancelled) setStatus(`Failed to load PDF: ${err.message}`);
      });
    return () => {
      cancelled = true;
      try {
        task.destroy?.();
      } catch (e) {
        /* noop */
      }
    };
  }, [docId]);

  // Render the highlighted pages whenever the highlight set (or doc) changes.
  useEffect(() => {
    let cancelled = false;
    const pdf = docRef.current;
    if (!pdf || !highlightPages || highlightPages.length === 0) {
      setRenderedPages([]);
      return;
    }
    const pages = [...highlightPages].sort((a, b) => a.page - b.page);

    (async () => {
      const done = [];
      for (const { page } of pages) {
        if (cancelled) return;
        const pdfPageNumber = page + 1; // datalab pages are 0-indexed
        if (pdfPageNumber < 1 || pdfPageNumber > pdf.numPages) continue;
        try {
          const pdfPage = await pdf.getPage(pdfPageNumber);
          if (cancelled) return;
          const base = pdfPage.getViewport({ scale: 1 });
          const scale = RENDER_WIDTH / base.width;
          const viewport = pdfPage.getViewport({ scale });
          const canvas = document.createElement("canvas");
          canvas.width = Math.ceil(viewport.width);
          canvas.height = Math.ceil(viewport.height);
          const ctx = canvas.getContext("2d");
          await pdfPage.render({ canvasContext: ctx, viewport }).promise;
          if (cancelled) return;
          done.push({
            page,
            dataUrl: canvas.toDataURL("image/png"),
            width: canvas.width,
            height: canvas.height,
          });
          setRenderedPages([...done]);
        } catch (e) {
          // skip a page that fails to render
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [highlightPages, docToken]);

  // Scroll to the page containing the selected chunk.
  useEffect(() => {
    if (!selectedChunkId || !highlightPages) return;
    const target = highlightPages.find((p) => p.rects.some((r) => r.chunk_id === selectedChunkId));
    if (target && pageRefs.current[target.page]) {
      pageRefs.current[target.page].scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [selectedChunkId, highlightPages]);

  const rectsByPage = {};
  (highlightPages || []).forEach((p) => {
    rectsByPage[p.page] = p.rects;
  });

  return (
    <div className="pdf-viewer" ref={containerRef}>
      <div className="pdf-header">
        <span>PDF view</span>
        {highlightPages?.length > 0 && (
          <span className="subtle">{highlightPages.length} highlighted page(s)</span>
        )}
      </div>

      {status && <div className="pdf-status">{status}</div>}
      {!docId && <div className="pdf-status">Select a document to preview.</div>}
      {docId && !status && (!highlightPages || highlightPages.length === 0) && (
        <div className="pdf-status">Run a query to highlight matched chunks here.</div>
      )}

      <div className="pdf-pages">
        {renderedPages.map((rp) => (
          <div
            key={rp.page}
            className="pdf-page"
            ref={(el) => (pageRefs.current[rp.page] = el)}
            style={{ width: rp.width }}
          >
            <div className="pdf-page-label">Page {rp.page + 1}</div>
            <div className="pdf-page-canvas" style={{ width: rp.width, height: rp.height }}>
              <img src={rp.dataUrl} width={rp.width} height={rp.height} alt={`page ${rp.page + 1}`} />
              {(rectsByPage[rp.page] || []).map((r, i) => {
                const dim = selectedChunkId && r.chunk_id !== selectedChunkId;
                const active = selectedChunkId && r.chunk_id === selectedChunkId;
                return (
                  <div
                    key={i}
                    className={`hl ${active ? "active" : ""} ${dim ? "dim" : ""}`}
                    style={{
                      left: `${r.x * 100}%`,
                      top: `${r.y * 100}%`,
                      width: `${r.w * 100}%`,
                      height: `${r.h * 100}%`,
                    }}
                    title={`#${r.rank} score ${r.score}`}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
