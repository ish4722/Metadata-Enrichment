"""
Generate RAG queries from JSON schema keys using LLM.
This script analyzes a PDF document and converts JSON key paths into
contextual queries that can be used for RAG (Retrieval-Augmented Generation).
"""

import os
import re
import json
import pdfplumber
from typing import List, Dict
from langchain_openai import ChatOpenAI
from pathlib import Path
from sauth_token import get_token


# Configuration
PDF_PATH = r"c:\Users\ITandon2\Metadata-Enrichment\PR3216.pdf"
SCHEMA_PATH = r"c:\Users\ITandon2\Metadata-Enrichment\all_schema_keys.json"
OUTPUT_QUERIES_PATH = r"c:\Users\ITandon2\Metadata-Enrichment\rag_queries.txt"
BATCH_SIZE = 10  # Process keys in batches to reduce API calls
USE_LLM = True  # Set to True only if you want LLM refinement; deterministic mode is generic and safer


def remove_bearer(token: str) -> str:
    """Remove 'Bearer ' prefix from token if present."""
    return token[len("Bearer "):] if token.startswith("Bearer ") else token


def get_llm() -> ChatOpenAI:
    """Initialize and return the LLM client using the same config as rag_pdf_search.py"""
    token = remove_bearer(get_token())
    return ChatOpenAI(
        model="gpt-5o",
        base_url="https://datascience-services.app.weu-d1.delfi.slb-ds.com/ds/modelops/deployments/v1",
        api_key=token,
        default_headers={"slb-partition-id": "sandbox-weu-des-prod-testing-e"},
    )


def extract_pdf_context(pdf_path: str, max_pages: int = 5) -> str:
    """
    Return a generic description of oil and gas document structure.
    Use the reference PDF only to inform the fact that the document is a
    petroleum report; do not preserve PDF-specific names or identifiers.
    """
    return (
        "This document is a petroleum industry report containing typical sections "
        "such as report metadata, operator and facility details, well and wellbore "
        "information, document identifiers, system metadata, and data collection or "
        "completion event details. The query generation should remain general and "
        "reusable across diverse oil and gas report PDFs."
    )


def load_schema_keys(schema_path: str) -> List[Dict]:
    """Load the schema keys from the JSON file."""
    print(f"Loading schema keys from: {schema_path}")
    
    with open(schema_path, 'r') as f:
        data = json.load(f)
    
    key_paths = data.get('key_paths', [])
    print(f"Loaded {len(key_paths)} key paths")
    
    return key_paths


def humanize_key_path(key_path: str) -> str:
    """Convert JSON key path into a generic human-readable search query."""
    replacements = {
        'acl': 'access control list',
        'uwi': 'unique well identifier',
        'ubhi': 'unique borehole identifier',
        'tvd': 'true vertical depth',
        'wgs84': 'WGS84 coordinates',
        'slb': 'SLB',
        'pml': 'permit',
        'documnet': 'document',
        'uwi': 'unique well identifier',
        'ubhi': 'unique borehole identifier',
        'id': 'ID',
        'ids': 'IDs',
    }

    def split_camel_case(text: str) -> str:
        text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
        return text

    parts = re.split(r'[._]', key_path)
    tokens = []

    for part in parts:
        part = split_camel_case(part)
        part = part.replace('-', ' ').strip().lower()
        if not part or part in {'data', 'metadata'}:
            continue

        for word in part.split():
            if word in replacements:
                tokens.append(replacements[word])
            elif word.endswith('id') and word != 'id':
                tokens.append(f"{word[:-2].strip()} ID")
            elif word == 'id':
                tokens.append('ID')
            else:
                tokens.append(word)

    phrase = ' '.join(tokens)
    phrase = re.sub(r'\s+', ' ', phrase).strip()
    if not phrase:
        phrase = key_path

    return f"What information is available about {phrase}?"


def is_generic_query(query: str) -> bool:
    """Return False if the query contains PDF-specific or unique text."""
    banned = [
        'ahuroa',
        'swift energy',
        'tawn',
        'pr3216',
        'new zealand',
        'pml',
        '20260326',
        '20260327',
        'collection id',
    ]
    lowered = query.lower()
    return not any(token in lowered for token in banned)


def generate_queries_batch(
    llm: ChatOpenAI,
    key_path_objects: List[Dict],
    pdf_context: str
) -> List[str]:
    """
    Generate multiple generic queries for each key path.
    """
    key_paths = [obj['key_path'] for obj in key_path_objects]
    fallback_queries = [humanize_key_path(key) for key in key_paths]

    if not USE_LLM:
        return fallback_queries

    prompt = f"""You are an expert in oil & gas data retrieval and RAG systems.
Your task is to convert JSON schema key paths into effective, reusable search queries for petroleum documents.

DOCUMENT CONTEXT:
{pdf_context}

KEY PATHS TO CONVERT:
{json.dumps(key_paths, indent=2)}

Important:
- Do not use any specific report titles, company names, well names, region names, or unique identifiers.
- Keep queries general so they can apply across many oil and gas PDF reports.
- Focus on the semantic meaning of each key path.

Format your response as a numbered list where each line contains ONLY the query.
Generate exactly {len(key_paths)} queries."""

    try:
        response = llm.invoke(prompt)
        response_text = str(response.content) if hasattr(response, 'content') else str(response)

        queries = []
        for line in response_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                query = line.lstrip('0123456789.-) ').strip()
                if query and len(query) > 5:
                    queries.append(query)

        if len(queries) != len(key_paths) or any(not is_generic_query(q) for q in queries):
            print("  Warning: LLM response was not fully generic; using deterministic fallback for this batch.")
            return fallback_queries

        return queries
    except Exception as e:
        print(f"Error generating queries: {e}")
        return fallback_queries


def generate_all_queries(
    llm: ChatOpenAI,
    key_path_objects: List[Dict],
    pdf_context: str,
    batch_size: int = BATCH_SIZE
) -> List[str]:
    """
    Generate queries for all key paths, processing in batches.
    """
    all_queries = []
    total_keys = len(key_path_objects)
    
    print(f"\nGenerating queries for {total_keys} keys in batches of {batch_size}...")
    
    for i in range(0, total_keys, batch_size):
        batch = key_path_objects[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_keys + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} keys)...")
        
        queries = generate_queries_batch(llm, batch, pdf_context)
        all_queries.extend(queries)
        
        print(f"  Generated {len(queries)} queries")
    
    return all_queries


def save_queries_to_file(queries: List[str], output_path: str) -> None:
    """
    Save all generated queries to a text file, one per line.
    """
    print(f"\nSaving {len(queries)} queries to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# RAG Queries Generated from JSON Schema Keys\n")
        f.write(f"# Total queries: {len(queries)}\n")
        f.write("# Format: One query per line\n")
        f.write("# Usage: Use these queries to search and retrieve relevant data from PDF documents\n\n")
        
        for i, query in enumerate(queries, 1):
            f.write(f"{i}. {query}\n")
    
    print("[OK] Queries saved successfully!")
    print(f"  File: {output_path}")


def analyze_pdf_structure(pdf_path: str) -> None:
    """
    Analyze and print basic statistics about the PDF.
    """
    print(f"\n{'='*60}")
    print("PDF STRUCTURE ANALYSIS")
    print(f"{'='*60}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Total pages: {len(pdf.pages)}")
            
            # Analyze first page
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            tables = first_page.extract_tables()
            
            print(f"First page text length: {len(text) if text else 0} characters")
            print(f"First page tables: {len(tables) if tables else 0}")
            
            if text:
                # Show first 500 characters
                print(f"\nFirst 300 characters of PDF:")
                print("-" * 60)
                print(text[:300])
                print("-" * 60)
            
            if tables and len(tables) > 0:
                print(f"\nFirst table columns: {len(tables[0][0]) if tables[0] else 0}")
                if len(tables[0]) > 0:
                    headers = tables[0][0]
                    print(f"Column names: {headers[:5]}")  # Show first 5 columns
    
    except Exception as e:
        print(f"Error analyzing PDF: {e}")
    
    print(f"{'='*60}\n")


def main():
    """Main execution function."""
    
    print("\n" + "="*60)
    print("RAG QUERY GENERATOR - JSON Schema to LLM Queries")
    print("="*60)
    
    # Check if files exist
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: PDF file not found: {PDF_PATH}")
        print("   Looking for available PDFs...")
        metadata_dir = os.path.dirname(PDF_PATH)
        pdfs = list(Path(metadata_dir).glob("*.pdf"))
        if pdfs:
            print(f"   Found: {[p.name for p in pdfs]}")
        return
    
    if not os.path.exists(SCHEMA_PATH):
        print(f"❌ Error: Schema file not found: {SCHEMA_PATH}")
        return
    
    # Analyze PDF structure
    analyze_pdf_structure(PDF_PATH)
    
    # Initialize LLM
    print("Initializing LLM...")
    llm = get_llm()
    print("[OK] LLM initialized")
    
    # Extract PDF context
    pdf_context = extract_pdf_context(PDF_PATH, max_pages=3)
    print(f"[OK] Extracted PDF context ({len(pdf_context)} characters)")
    
    # Load schema keys
    key_path_objects = load_schema_keys(SCHEMA_PATH)
    
    # Generate queries
    queries = generate_all_queries(llm, key_path_objects, pdf_context, batch_size=BATCH_SIZE)
    
    # Save queries
    save_queries_to_file(queries, OUTPUT_QUERIES_PATH)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries generated: {len(queries)}")
    print(f"Output file: {OUTPUT_QUERIES_PATH}")
    print(f"{'='*60}\n")
    
    # Show sample queries
    if queries:
        print("Sample queries (first 5):")
        for i, query in enumerate(queries[:5], 1):
            print(f"  {i}. {query}")


if __name__ == "__main__":
    main()
