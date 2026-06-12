"""
Convert extracted PDF JSON into metadata-rich chunks and apply a RAG pipeline.

This script:
1. Loads OCR/extracted JSON output from your PDF extraction tool.
2. Converts HTML blocks into plain text chunks with metadata.
3. Groups and chunks text to create useful retrieval units.
4. Builds a retrieval index using TF-IDF embeddings.
5. Supports query-time retrieval and optional LLM answer generation.
   - After retrieval, the FULL PAGE content of each matched chunk is sent to the LLM.
   - A file is written containing all pages (and their content) sent to the LLM.

Environment Variables:
  LLM_MODEL              - LLM model name (default: gpt-5-mini)
  LLM_BASE_URL           - LLM service base URL
  SLB_PARTITION_ID       - SLB partition ID for API headers

Copy .env.example to .env and update with your credentials.

Usage examples:
  python datalab_rag_pipeline.py build \
    --input "datalab-output-PR3216.pdf (1).json" \
    --chunks-output useful_chunks.jsonl \
    --index-output rag_index_iteration1.pkl

  python datalab_rag_pipeline.py query \
    --index-output rag_index_iteration1.pkl \
    --query "What is the operator name?" \
    --top-k 10
"""

import argparse
import html
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dotenv import load_dotenv
except ImportError:
    dotenv_available = False
else:
    dotenv_available = True
    load_dotenv()

try:
    from langchain_openai import ChatOpenAI
    from sauth_token import get_token
    HAS_LLM = True
except Exception:
    HAS_LLM = False


@dataclass
class Chunk:
    id: str
    source: str
    page: int
    section_path: str
    block_type: str
    text: str
    metadata: Dict[str, Any]


def html_to_text(raw_html: str) -> str:
    raw_html = html.unescape(raw_html or "")
    raw_html = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    raw_html = re.sub(r"<style.*?>.*?</style>", "", raw_html, flags=re.IGNORECASE | re.DOTALL)
    raw_html = re.sub(r"<script.*?>.*?</script>", "", raw_html, flags=re.IGNORECASE | re.DOTALL)
    raw_html = re.sub(r"<.*?>", "", raw_html)
    raw_html = re.sub(r"\s+", " ", raw_html)
    return raw_html.strip()


def normalize_section_path(section_titles: List[str]) -> str:
    return " > ".join([title.strip() for title in section_titles if title and title.strip()])


def flatten_blocks(node: Dict[str, Any], page: int = -1, section_titles: Optional[List[str]] = None) -> List[Chunk]:
    if section_titles is None:
        section_titles = []

    blocks: List[Chunk] = []
    node_page = node.get("page", page)
    node_block_type = node.get("block_type", "Unknown")
    node_id = node.get("id", "")
    node_html = node.get("html", "")

    if node_block_type == "SectionHeader" and node_html:
        heading_text = html_to_text(node_html)
        if heading_text:
            section_titles = section_titles + [heading_text]

    if node_block_type in {"Text", "SectionHeader", "Table", "PageHeader"} and node_html:
        text = html_to_text(node_html)
        if text:
            chunk = Chunk(
                id=node_id,
                source="",
                page=int(node_page) if isinstance(node_page, int) else -1,
                section_path=normalize_section_path(section_titles),
                block_type=node_block_type,
                text=text,
                metadata={
                    "block_id": node_id,
                    "block_type": node_block_type,
                    "page": int(node_page) if isinstance(node_page, int) else -1,
                    "bbox": node.get("bbox"),
                    "polygon": node.get("polygon"),
                    "section_hierarchy": node.get("section_hierarchy"),
                    "raw_html": node_html,
                },
            )
            blocks.append(chunk)

    for child in node.get("children", []):
        blocks.extend(flatten_blocks(child, page=node_page, section_titles=section_titles))

    return blocks


def load_extracted_json(path: str) -> List[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("children", [])
    chunks: List[Chunk] = []

    for page_node in pages:
        source = Path(path).name
        page_index = page_node.get("page", -1)
        page_chunks = flatten_blocks(page_node, page=page_index, section_titles=[])
        for chunk in page_chunks:
            chunk.source = source
            chunks.append(chunk)

    return chunks


def merge_blocks_by_section(chunks: List[Chunk]) -> List[Chunk]:
    grouped: Dict[Tuple[str, int, str], List[Chunk]] = {}
    for chunk in chunks:
        key = (chunk.source, chunk.page, chunk.section_path or chunk.block_type)
        grouped.setdefault(key, []).append(chunk)

    merged: List[Chunk] = []
    for key, group in grouped.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        texts = [c.text for c in group if c.text.strip()]
        merged_text = "\n\n".join(texts).strip()
        merged_metadata = {
            "source": group[0].source,
            "page": group[0].page,
            "section_path": group[0].section_path,
            "block_type": group[0].block_type,
            "block_ids": [c.id for c in group],
            "raw_html_blocks": [c.metadata.get("raw_html") for c in group],
        }
        merged.append(
            Chunk(
                id=f"{group[0].source}::{group[0].page}::{len(merged)}",
                source=group[0].source,
                page=group[0].page,
                section_path=group[0].section_path,
                block_type=group[0].block_type,
                text=merged_text,
                metadata=merged_metadata,
            )
        )

    return merged


def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        window = words[start:end]
        chunks.append(" ".join(window).strip())
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def build_useful_chunks(chunks: List[Chunk], chunk_size: int = 800, overlap: int = 200) -> List[Chunk]:
    useful_chunks: List[Chunk] = []
    id_counter = 0

    for chunk in chunks:
        text_chunks = split_text_into_chunks(chunk.text, chunk_size=chunk_size, overlap=overlap)
        for part in text_chunks:
            if not part.strip():
                continue
            useful_chunks.append(
                Chunk(
                    id=f"chunk_{id_counter}",
                    source=chunk.source,
                    page=chunk.page,
                    section_path=chunk.section_path,
                    block_type=chunk.block_type,
                    text=part,
                    metadata={
                        **chunk.metadata,
                        "chunk_id": f"chunk_{id_counter}",
                        "source": chunk.source,
                        "page": chunk.page,
                        "section_path": chunk.section_path,
                        "block_type": chunk.block_type,
                        "text_length": len(part),
                    },
                )
            )
            id_counter += 1

    return useful_chunks


def save_chunks_jsonl(chunks: List[Chunk], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "id": chunk.id,
                "source": chunk.source,
                "page": chunk.page,
                "section_path": chunk.section_path,
                "block_type": chunk.block_type,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chunks_jsonl(path: str) -> List[Chunk]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            chunks.append(
                Chunk(
                    id=data["id"],
                    source=data["source"],
                    page=data["page"],
                    section_path=data.get("section_path", ""),
                    block_type=data.get("block_type", ""),
                    text=data["text"],
                    metadata=data.get("metadata", {}),
                )
            )
    return chunks


def build_vector_store(chunks: List[Chunk]) -> Tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(stop_words="english")
    texts = [chunk.text for chunk in chunks]
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def save_index(index_path: str, chunks: List[Chunk], vectorizer: TfidfVectorizer, matrix: np.ndarray) -> None:
    with open(index_path, "wb") as f:
        pickle.dump({"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}, f)


def load_index(index_path: str) -> Tuple[List[Chunk], TfidfVectorizer, np.ndarray]:
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return data["chunks"], data["vectorizer"], data["matrix"]


def retrieve_top_chunks(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: np.ndarray,
    chunks: List[Chunk],
    top_k: int = 5,
) -> List[Tuple[Chunk, float]]:
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]
    indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in indices if scores[i] > 0]


# ---------------------------------------------------------------------------
# Full-page expansion helpers
# ---------------------------------------------------------------------------

def build_page_index(chunks: List[Chunk]) -> Dict[Tuple[str, int], List[Chunk]]:
    """Build a mapping of (source, page) -> all chunks on that page."""
    page_index: Dict[Tuple[str, int], List[Chunk]] = {}
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        page_index.setdefault(key, []).append(chunk)
    return page_index


def get_full_pages_for_matches(
    matches: List[Tuple[Chunk, float]],
    page_index: Dict[Tuple[str, int], List[Chunk]],
) -> List[Dict[str, Any]]:
    """
    Given a list of matched (chunk, score) pairs, return deduplicated full-page
    content objects ordered by (source, page).

    Each object has:
      - source, page, matched_score (best score among chunks on that page),
        matched_chunk_ids, full_text (all chunks on the page joined together)
    """
    # Track the best score seen for each page
    page_best_score: Dict[Tuple[str, int], float] = {}
    page_matched_ids: Dict[Tuple[str, int], List[str]] = {}

    for chunk, score in matches:
        key = (chunk.source, chunk.page)
        if key not in page_best_score or score > page_best_score[key]:
            page_best_score[key] = score
        page_matched_ids.setdefault(key, []).append(chunk.id)

    full_pages: List[Dict[str, Any]] = []
    for key in sorted(page_best_score.keys()):
        source, page = key
        page_chunks = page_index.get(key, [])
        # Sort chunks by their natural order (id is chunk_<int> so sort numerically where possible)
        page_chunks_sorted = sorted(
            page_chunks,
            key=lambda c: int(c.id.split("_")[-1]) if c.id.startswith("chunk_") else 0,
        )
        full_text = "\n\n".join(c.text for c in page_chunks_sorted if c.text.strip())
        full_pages.append(
            {
                "source": source,
                "page": page,
                "matched_score": page_best_score[key],
                "matched_chunk_ids": page_matched_ids[key],
                "num_chunks_on_page": len(page_chunks),
                "full_text": full_text,
            }
        )

    return full_pages


def save_pages_sent_to_llm(full_pages: List[Dict[str, Any]], query: str, output_path: str) -> None:
    """Write a JSON file listing every page (and its full text) that was sent to the LLM."""
    payload = {
        "query": query,
        "num_pages": len(full_pages),
        "pages": full_pages,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Pages sent to LLM written to: {output_path}")


# ---------------------------------------------------------------------------
# Prompt / LLM helpers  (updated to accept full pages instead of chunks)
# ---------------------------------------------------------------------------

def make_prompt_from_pages(
    query: str,
    full_pages: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    history_text = ""
    if chat_history:
        history_lines = []
        for turn in chat_history:
            history_lines.append(f"Previous question: {turn['query']}")
            history_lines.append(f"Previous answer: {turn['answer']}")
        history_text = "\n".join(history_lines) + "\n\n"

    context_parts = []
    for page_obj in full_pages:
        header = f"[source={page_obj['source']}, page={page_obj['page']}]"
        context_parts.append(f"{header}\n{page_obj['full_text']}")
    context = "\n\n---\n\n".join(context_parts)

    return (
        "You are an expert retrieval assistant for oil and gas reports. "
        "Use only the provided page excerpts and include the source page for each answer. "
        "If the answer is not contained in the excerpts, say that the information is unavailable in the document.\n\n"
        f"{history_text}"
        f"Question: {query}\n\n"
        "Relevant pages:\n"
        f"{context}\n"
        "Answer concisely and cite the source page when possible."
    )


def get_llm() -> Optional["ChatOpenAI"]:
    if not HAS_LLM:
        return None
    token = get_token()
    if token.startswith("Bearer "):
        token = token[len("Bearer "):]

    model = os.getenv("LLM_MODEL", "gpt-5.2")
    base_url = os.getenv(
        "LLM_BASE_URL",
        "https://datascience-services.app.weu-d1.delfi.slb-ds.com/ds/modelops/deployments/v1",
    )
    partition_id = os.getenv("SLB_PARTITION_ID", "sandbox-weu-des-prod-testing-e")

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=token,
        default_headers={"slb-partition-id": partition_id},
    )


def answer_with_llm(
    query: str,
    full_pages: List[Dict[str, Any]],
    llm: "ChatOpenAI",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    prompt = make_prompt_from_pages(query, full_pages, chat_history=chat_history)
    response = llm.invoke(prompt)
    return str(response.content) if hasattr(response, "content") else str(response)


# ---------------------------------------------------------------------------
# Pipeline entry-points
# ---------------------------------------------------------------------------

def interactive_query_loop(index_path: str, top_k: int) -> None:
    print(f"Loading retrieval index from {index_path}")
    chunks, vectorizer, matrix = load_index(index_path)
    page_index = build_page_index(chunks)
    chat_history: List[Dict[str, str]] = []
    print("Entering interactive mode. Type 'exit' or blank line to quit.")

    while True:
        query = input("Query: ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            print("Exiting interactive mode.")
            break

        matches = retrieve_top_chunks(query, vectorizer, matrix, chunks, top_k=top_k)
        if not matches:
            print("No matching chunks found.")
            continue

        # ------------------------------------------------------------------
        # Expand matched chunks → full pages
        # ------------------------------------------------------------------
        full_pages = get_full_pages_for_matches(matches, page_index)

        print(f"\nMatched {len(matches)} chunks across {len(full_pages)} unique page(s).")
        for idx, (chunk, score) in enumerate(matches, 1):
            print(
                f"  [{idx}] score={score:.4f} source={chunk.source} "
                f"page={chunk.page} section={chunk.section_path or chunk.block_type}"
            )

        print(f"\nFull pages being sent to LLM:")
        for page_obj in full_pages:
            print(
                f"  page={page_obj['page']} source={page_obj['source']} "
                f"(best_score={page_obj['matched_score']:.4f}, "
                f"{page_obj['num_chunks_on_page']} chunks on page)"
            )
            print(page_obj["full_text"][:600].strip(), "...\n")

        # Save pages-sent-to-LLM file
        safe_query = re.sub(r"[^\w\s-]", "", query)[:50].strip().replace(" ", "_")
        pages_output_path = f"pages_sent_to_llm_{safe_query}.json"
        save_pages_sent_to_llm(full_pages, query, pages_output_path)

        if HAS_LLM:
            print("\nGenerating final answer with LLM...")
            llm = get_llm()
            if llm is not None:
                answer = answer_with_llm(query, full_pages, llm, chat_history=chat_history)
                print("\nLLM answer:\n", answer)
                chat_history.append({"query": query, "answer": answer})
            else:
                print("LLM support is not available in this environment.")
        else:
            print("\nLLM support is not available in this environment.")


def build_pipeline(
    input_path: str,
    chunks_output: str,
    index_output: str,
    chunk_size: int,
    overlap: int,
) -> None:
    print(f"Loading extracted JSON from {input_path}")
    base_chunks = load_extracted_json(input_path)
    print(f"Extracted {len(base_chunks)} base text blocks")

    print("Merging blocks by page and section")
    merged_chunks = merge_blocks_by_section(base_chunks)
    print(f"Merged into {len(merged_chunks)} section-level chunks")

    print("Chunking long blocks into retrieval units")
    useful_chunks = build_useful_chunks(merged_chunks, chunk_size=chunk_size, overlap=overlap)
    print(f"Created {len(useful_chunks)} useful retrieval chunks")

    print(f"Saving chunks to {chunks_output}")
    save_chunks_jsonl(useful_chunks, chunks_output)

    print("Building vector index for retrieval")
    vectorizer, matrix = build_vector_store(useful_chunks)
    save_index(index_output, useful_chunks, vectorizer, matrix)
    print(f"Saved retrieval index to {index_output}")


def query_pipeline(index_path: str, query: str, top_k: int) -> None:
    print(f"Loading retrieval index from {index_path}")
    chunks, vectorizer, matrix = load_index(index_path)
    page_index = build_page_index(chunks)

    matches = retrieve_top_chunks(query, vectorizer, matrix, chunks, top_k=top_k)
    if not matches:
        print("No matching chunks found.")
        return

    # ------------------------------------------------------------------
    # Expand matched chunks → full pages
    # ------------------------------------------------------------------
    full_pages = get_full_pages_for_matches(matches, page_index)

    print(f"\nMatched {len(matches)} chunks across {len(full_pages)} unique page(s).")
    for idx, (chunk, score) in enumerate(matches, 1):
        print(
            f"  [{idx}] score={score:.4f} source={chunk.source} "
            f"page={chunk.page} section={chunk.section_path or chunk.block_type}"
        )

    print(f"\nFull pages being sent to LLM:")
    for page_obj in full_pages:
        print(
            f"  page={page_obj['page']} source={page_obj['source']} "
            f"(best_score={page_obj['matched_score']:.4f}, "
            f"{page_obj['num_chunks_on_page']} chunks on page)"
        )
        print(page_obj["full_text"][:600].strip(), "...\n")

    # Save pages-sent-to-LLM file
    safe_query = re.sub(r"[^\w\s-]", "", query)[:50].strip().replace(" ", "_")
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent

    queries_dir = BASE_DIR / "queries"
    queries_dir.mkdir(exist_ok=True)

    pages_output_path = queries_dir / f"pages_sent_to_llm_{safe_query}.json"

    save_pages_sent_to_llm(full_pages, query, str(pages_output_path))

    if HAS_LLM:
        print("\nGenerating final answer with LLM...")
        llm = get_llm()
        if llm is not None:
            answer = answer_with_llm(query, full_pages, llm)
            print("\nLLM answer:\n", answer)
        else:
            print("LLM support is not available in this environment.")
    else:
        print("\nLLM support is not available in this environment.")


# ---------------------------------------------------------------------------
# Argument parsing  (fixed)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    if len(sys.argv) == 1:
        return argparse.Namespace(
            command="interactive",
            index_output="rag_index_iteration1.pkl",
            top_k=10,
        )

    parser = argparse.ArgumentParser(
        description="Build and query a RAG pipeline from extracted PDF JSON."
    )
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser("build", help="Build chunks and index from extracted JSON")
    build.add_argument("--input", required=True, help="Path to extracted JSON file")
    build.add_argument("--chunks-output", default="useful_chunks.jsonl", help="Output JSONL file for chunks")
    build.add_argument("--index-output", default="rag_index_iteration1.pkl", help="Output pickle file for retrieval index")
    build.add_argument("--chunk-size", type=int, default=800, help="Maximum chunk size in words")
    build.add_argument("--overlap", type=int, default=200, help="Chunk overlap in words")

    qp = subparsers.add_parser("query", help="Query the retrieval index")
    qp.add_argument("--index-output", default="rag_index_iteration1.pkl", help="Path to retrieval index pickle")
    qp.add_argument("query", help="Search query")
    qp.add_argument("--top-k", type=int, default=10, help="Number of top chunks to retrieve")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "build":
        build_pipeline(
            input_path=args.input,
            chunks_output=args.chunks_output,
            index_output=args.index_output,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    elif args.command == "query":
        query_pipeline(
            index_path=args.index_output,
            query=args.query,
            top_k=args.top_k,
        )
    elif args.command == "interactive":
        interactive_query_loop(
            index_path=args.index_output,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()