#convert ocr json output into langchain docs
import json
import html

from bs4 import BeautifulSoup
from langchain_core.documents import Document


#convert html to clean text
def html_to_text(raw_html):
    """
    Converts OCR HTML into readable text.
    """

    # Decode escaped HTML entities
    decoded_html = html.unescape(raw_html)

    # Parse HTML
    soup = BeautifulSoup(decoded_html, "html.parser")

    # Extract readable text
    text = soup.get_text(separator="\n")

    # Clean whitespace
    return text.strip()


# ============================================================
# Main Conversion Function
# ============================================================

def convert_ocr_json_to_langchain_docs(json_path, source_name):
    """
    Converts OCR JSON into LangChain Documents.

    Parameters
    ----------
    json_path : str
        Path to OCR JSON file

    source_name : str
        Name of source PDF/document

    Returns
    -------
    list[Document]
    """

    # ------------------------------------------------
    # Load OCR JSON
    # ------------------------------------------------

    with open(json_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    # Final list of LangChain documents
    documents = []

    for page in ocr_data["children"]: #goes through all the pages

        page_number = page.get("page", -1)

        page_chunks = page.get("children", [])   #lists all chunks in given page

        for chunk_index, chunk in enumerate(page_chunks):
            #pull out all relevant data
            chunk_id = chunk.get("id")
            chunk_type = chunk.get("block_type")
            raw_html = chunk.get("html", "")
            bbox = chunk.get("bbox")
            polygon = chunk.get("polygon")

            section_hierarchy = chunk.get("section_hierarchy", {})
            
            #skip empty chunkd
            if not raw_html.strip():
                continue

            #covert html content to plain text
            cleaned_text = html_to_text(raw_html)
            # Skip empty results
            if not cleaned_text:
                continue

            #creat langchain doc
            doc = Document(
                # Main chunk text
                page_content=cleaned_text,
                # Metadata
                metadata={
                    "source": source_name,
                    "page": page_number,
                    "chunk_id": chunk_id, #og ocr id
                    "chunk_type": chunk_type, #modality
                    "page_chunk_index": chunk_index, #pos in page
                    # Layout
                    "bbox": bbox,
                    "polygon": polygon,
                    # Hierarchy information
                    "section_hierarchy": section_hierarchy,
                    # Preserve original html
                    "raw_html": raw_html
                }
            )
            documents.append(doc)

    return documents


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------
    # Convert OCR JSON
    # --------------------------------------------

    docs = convert_ocr_json_to_langchain_docs(
    json_path="trial_json_files/datalab-output-PR3216.pdf.json",
    source_name="datalab-output-PR3216.pdf"
)


    # --------------------------------------------
    # Print Results
    # --------------------------------------------

    print("\nTOTAL DOCUMENTS:")
    print(len(docs))

    print("\nFIRST DOCUMENT:\n")

    print(docs[0])

    print("\nPAGE CONTENT:\n")
    print(docs[0].page_content)

    print("\nMETADATA:\n")
    print(docs[0].metadata)