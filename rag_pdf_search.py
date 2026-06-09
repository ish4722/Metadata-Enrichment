import os
import pickle
from typing import List, Tuple
import numpy as np
import pdfplumber
from langchain_openai import ChatOpenAI
from sauth_token import get_token
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "New_Zealand_Petroleum_Basins_2014-2015_Revised.pdf",
)
CACHE_PATH = os.path.join(
    os.path.dirname(__file__),
    "pdf_chunks_cache.pkl",
)


def remove_bearer(token: str) -> str:
    return token[len("Bearer "):] if token.startswith("Bearer ") else token


def load_pdf_chunks(pdf_path: str, chunk_size: int = 900, overlap: int = 200) -> List[dict]:
    """Load a PDF and split it into text chunks with lightweight table/image notes."""
    chunks = []
    page_texts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            page_content = []

            if page_text.strip():
                page_content.append(f"Page {page_number} text:\n{page_text}")

            tables = page.extract_tables() or []
            for table_index, table in enumerate(tables, start=1):
                rows = [" | ".join([cell or "" for cell in row]) for row in table]
                table_text = "\n".join(rows)
                page_content.append(f"Page {page_number} table {table_index}:\n{table_text}")

            if page.images:
                page_content.append(
                    f"Page {page_number} contains {len(page.images)} image(s). "
                    "Visual content is noted, but not OCRed in this basic pipeline."
                )

            if page_content:
                page_texts.append("\n\n".join(page_content))

    combined_text = "\n\n".join(page_texts)
    tokens = combined_text.split()
    start = 0
    chunk_id = 1

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        text_chunk = " ".join(chunk_tokens)
        chunks.append({
            "id": chunk_id,
            "content": text_chunk,
        })
        chunk_id += 1
        start += chunk_size - overlap

    return chunks


def build_vector_store(chunks: List[dict]) -> Tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(stop_words="english")
    docs = [chunk["content"] for chunk in chunks]
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


def load_or_build_cache(pdf_path: str, chunk_size: int = 900, overlap: int = 200):
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            return cache["chunks"], cache["vectorizer"], cache["matrix"]
        except Exception:
            pass

    chunks = load_pdf_chunks(pdf_path, chunk_size, overlap)
    vectorizer, matrix = build_vector_store(chunks)
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}, f)
    except Exception:
        pass
    return chunks, vectorizer, matrix


def retrieve_top_chunks(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: np.ndarray,
    chunks: List[dict],
    top_k: int = 4,
) -> List[dict]:
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append(
            {
                "id": chunks[idx]["id"],
                "content": chunks[idx]["content"],
                "score": float(scores[idx]),
            }
        )
    return results


def make_prompt(query: str, chunks: List[dict]) -> str:
    context = "\n\n---\n\n".join(
        [f"Chunk {chunk['id']}:\n{chunk['content']}" for chunk in chunks]
    )
    prompt = (
        "You are a oil and gas industry expert that answers questions using the provided document excerpts. "
        "If the answer is not in the text, say that the information is unavailable in the document.\n\n"
        "Document excerpts:\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        "Answer concisely and cite the relevant chunk content when possible."
    )
    return prompt


def get_llm() -> ChatOpenAI:
    token = remove_bearer(get_token())
    return ChatOpenAI(
        model="gpt-4o",
        base_url="https://datascience-services.app.weu-d1.delfi.slb-ds.com/ds/modelops/deployments/v1",
        api_key=token,
        default_headers={"slb-partition-id": "sandbox-weu-des-prod-testing-e"},
    )


def answer_query(query: str, vectorizer, matrix, chunks, llm) -> str:
    if not query or not query.strip():
        return "Please enter a query to search the PDF."

    top_chunks = retrieve_top_chunks(query, vectorizer, matrix, chunks)
    if not top_chunks:
        return "No matching passages found in the document."

    prompt = make_prompt(query, top_chunks)

    try:
        response = llm.invoke(prompt)
        answer = response
    except Exception as exc:
        answer = (
            "LLM call failed, returning the most relevant document excerpts instead.\n\n"
            + "\n\n---\n\n".join(
                [f"Chunk {chunk['id']}: {chunk['content']}" for chunk in top_chunks]
            )
        )

    return answer


def create_app():
    # CLI runner: loads cache and enters a read-eval loop to answer queries
    chunks, vectorizer, matrix = load_or_build_cache(PDF_PATH)
    llm = get_llm()

    def cli_loop():
        print("PDF RAG Search (terminal mode). Type 'exit' to quit.")
        while True:
            try:
                query = input("\nEnter your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting.")
                break

            top_chunks = retrieve_top_chunks(query, vectorizer, matrix, chunks, top_k=4)
            print("\nTop matched chunks:")
            for c in top_chunks:
                print(f"- Chunk {c['id']} (score={c['score']:.4f})\n  {c['content'][:500].replace('\n', ' ')}\n")

            print("Calling LLM for a concise answer...")
            answer = answer_query(query, vectorizer, matrix, chunks, llm)
            print("\nLLM Answer:\n", answer)

    return cli_loop


if __name__ == "__main__":
    app = create_app()
    try:
        # `create_app` returns a CLI function; call it to start the terminal loop
        app()
    except Exception as exc:
        print("Failed to run CLI app:", exc)
