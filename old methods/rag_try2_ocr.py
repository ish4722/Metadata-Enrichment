import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pdfplumber
from langchain_openai import ChatOpenAI
from sauth_token import get_token
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


'''
DOCUMENTS_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "documents",
)

CACHE_PATH = os.path.join(
    os.path.dirname(__file__),
   "multi_doc_cache.pkl",
)'''

def remove_bearer(token: str) -> str:
    return token[len("Bearer "):] if token.startswith("Bearer ") else token

'''
def load_pdf_chunks(pdf_path: str, chunk_size: int = 900, overlap: int = 200) -> List[dict]:
    chunks = []
    page_texts = []
    pdf_name = os.path.basename(pdf_path)

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
        end = min(start + chunk_size, len(tokens)) #prevents from going over end
        chunk_tokens = tokens[start:end]
        text_chunk = " ".join(chunk_tokens) #create word from sentence
        chunks.append({
            "id": chunk_id,
            "content": text_chunk,
            "source_doc": pdf_name, 
            "page": "unknown",  
            "modality": "text",
            "has_image":False,
        })
        chunk_id += 1
        start += chunk_size - overlap

    return chunks
    '''

'''
def load_all_pdf_chunks(documents_folder: str, chunk_size: int = 900, overlap: int = 200) -> List[dict]:
    all_chunks = []
    pdf_files = list(Path(documents_folder).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {documents_folder}")
        return []
    
    print(f"Found {len(pdf_files)} PDF file(s). Loading...")
    
    for pdf_path in pdf_files:
        print(f"  Loading {pdf_path.name}...")
        try:
            chunks = load_pdf_chunks(str(pdf_path), chunk_size, overlap)
            all_chunks.extend(chunks)
            print(f"    ✓ Loaded {len(chunks)} chunks")
        except Exception as e:
            print(f"    ✗ Error loading {pdf_path.name}: {e}")
    
    print(f"Total chunks loaded: {len(all_chunks)}")
    return all_chunks
    '''

#get all chunks as langchain document list
from json_chunk import convert_ocr_json_to_langchain_docs

docs = convert_ocr_json_to_langchain_docs(
    json_path="trial_json_files/PR3216.json",
    source_name="PR3216.pdf"
)

print(len(docs))

#trial 1: convert langchain docs directly into full sized chunks (no rechunking)
def load_pdf_chunks(pdf_path: str, chunk_size: int = 900, overlap: int = 200) -> List[dict]:
    chunks = []
    page_texts = []
    pdf_name = os.path.basename(pdf_path)

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
        end = min(start + chunk_size, len(tokens)) #prevents from going over end
        chunk_tokens = tokens[start:end]
        text_chunk = " ".join(chunk_tokens) #create word from sentence
        chunks.append({
            "id": chunk_id,
            "content": text_chunk,
            "source_doc": pdf_name, 
            "page": "unknown",  
            "modality": "text",
            "has_image":False,
        })
        chunk_id += 1
        start += chunk_size - overlap

    return chunks

def build_vector_store(chunks: List[dict]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Build TF-IDF vectorizer for all chunks."""
    vectorizer = TfidfVectorizer(stop_words="english")
    docs = [chunk["content"] for chunk in chunks]
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


def load_or_build_cache(documents_folder: str, chunk_size: int = 900, overlap: int = 200):
    """Load from cache if available, otherwise build from scratch."""
    if os.path.exists(CACHE_PATH):
        try:
            print("Loading from cache...")
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            print(f"  ✓ Cache loaded ({len(cache['chunks'])} chunks)")
            return cache["chunks"], cache["vectorizer"], cache["matrix"]
        except Exception as e:
            print(f"  ✗ Cache load failed: {e}. Rebuilding...")

    chunks = load_all_pdf_chunks(documents_folder, chunk_size, overlap)
    if not chunks:
        print("ERROR: No chunks loaded. Check your documents folder.")
        return [], None, None
    
    print("Building vector store...")
    vectorizer, matrix = build_vector_store(chunks)
    print("  ✓ Vector store built")
    
    try:
        print("Saving to cache...")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}, f)
        print("  ✓ Cache saved")
    except Exception as e:
        print(f"  ✗ Cache save failed: {e}")
    
    return chunks, vectorizer, matrix


def retrieve_top_chunks(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: np.ndarray,
    chunks: List[dict],
    top_k: int = 4,
) -> List[dict]:
    """Retrieve top K chunks matching the query."""
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
                "source_doc": chunks[idx]["source_doc"],
            }
        )
    return results


def make_prompt(query: str, chunks: List[dict]) -> str:
    """Create prompt for LLM with retrieved chunks."""
    context = "\n\n---\n\n".join(
        [f"[From {chunk['source_doc']}] Chunk {chunk['id']}:\n{chunk['content']}" 
         for chunk in chunks]
    )
    prompt = (
        "You are a helpful assistant that answers questions using the provided document excerpts. "
        "If the answer is not in the text, say that the information is unavailable in the documents.\n\n"
        "Document excerpts:\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        "Answer concisely and cite the relevant document and chunk content when possible."
    )
    return prompt


def get_llm() -> ChatOpenAI:
    """Initialize LLM with company auth."""
    token = remove_bearer(get_token())
    return ChatOpenAI(
        model="gpt-4o",
        base_url="https://datascience-services.app.weu-d1.delfi.slb-ds.com/ds/modelops/deployments/v1",
        api_key=token,
        default_headers={"slb-partition-id": "sandbox-weu-des-prod-testing-e"},
    )


def answer_query(query: str, vectorizer, matrix, chunks, llm) -> str:
    """Main RAG function: retrieve relevant chunks and generate answer."""
    if not query or not query.strip():
        return "Please enter a query to search the documents."

    if vectorizer is None:
        return "ERROR: No documents loaded. Check your documents folder."

    top_chunks = retrieve_top_chunks(query, vectorizer, matrix, chunks)
    if not top_chunks:
        return "No matching passages found in the documents."

    prompt = make_prompt(query, top_chunks)

    try:
        response = llm.invoke(prompt)
        answer = response
    except Exception as exc:
        answer = (
            "LLM call failed, returning the most relevant document excerpts instead.\n\n"
            + "\n\n---\n\n".join(
                [f"[{chunk['source_doc']}] Chunk {chunk['id']}: {chunk['content'][:300]}" 
                 for chunk in top_chunks]
            )
        )

    return answer


def create_app():
    """Create and initialize the RAG app."""
    print("=" * 80)
    print("Multi-Document RAG Initialization")
    print("=" * 80)
    
    chunks, vectorizer, matrix = load_or_build_cache(DOCUMENTS_FOLDER)
    llm = get_llm()

    def cli_loop():
        """Interactive CLI for querying documents."""
        print("\n" + "=" * 80)
        print("Multi-Document RAG Search (terminal mode)")
        print("Type 'exit' to quit, 'stats' to see loaded documents")
        print("=" * 80)
        
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
            
            if query.lower() == "stats":
                doc_names = set(chunk["source_doc"] for chunk in chunks)
                print(f"\nLoaded {len(chunks)} chunks from {len(doc_names)} documents:")
                for doc in sorted(doc_names):
                    doc_chunks = sum(1 for c in chunks if c["source_doc"] == doc)
                    print(f"  - {doc}: {doc_chunks} chunks")
                continue

            top_chunks = retrieve_top_chunks(query, vectorizer, matrix, chunks, top_k=4)
            
            print("\nTop matched chunks:")
            for c in top_chunks:
                print(f"- [{c['source_doc']}] Chunk {c['id']} (score={c['score']:.4f})")
                print(f"  {c['content'][:400].replace(chr(10), ' ')}...\n")

            print("Calling LLM for a concise answer...")
            answer = answer_query(query, vectorizer, matrix, chunks, llm)
            print("\nLLM Answer:\n", answer)
            print("\n" + "-" * 80)

    return cli_loop


if __name__ == "__main__":
    app = create_app()
    try:
        app()
    except Exception as exc:
        print(f"Failed to run RAG app: {exc}")
        import traceback
        traceback.print_exc()