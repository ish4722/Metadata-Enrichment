"""
match_pdf_json.py
==================

Matches PDF files to their corresponding JSON files, using an Excel file
as the bridge between the two.

How it works
------------
1. EXCEL_FILE has a "CONFIDENTIAL" column (the pdf file name) and a
   "Well Name" column. Every PDF is looked up by file name in this
   table to find its Well Name.
2. Every JSON file in JSON_FOLDER is scanned recursively for a key that
   looks like "facility name" (facility_name, FacilityName, "Facility Name",
   etc. are all treated the same) and its value is read.
3. The Well Name (from step 1) and facility name (from step 2) are
   compared -- after normalising case/spacing/punctuation, with a fuzzy
   fallback for near-matches -- to decide which PDF and JSON belong together.
4. Matching JSON files are copied into MATCHED_JSON_FOLDER, renamed to
   <pdf-name>.json, and every pairing is written to PAIRS_LOG_FILE.

Before running
--------------
pip install pandas openpyxl
Then edit the CONFIGURATION block below to point at your real folders/file.
"""


"We did linking of pdfs from json then used json analyser to find out the missing keys and now we will make make schema for queries "
import os
import re
import json
import shutil
import difflib
import pandas as pd

# --------------------------------------------------------------------------
# CONFIGURATION -- edit these to match your setup
# --------------------------------------------------------------------------
PDF_FOLDER = r"C:\Users\ITandon2\Metadata-Enrichment\pdfs"                     # folder containing the .pdf files
JSON_FOLDER = r"C:\Users\ITandon2\Downloads\master-data--Wellbore\master-data--Wellbore"             # folder containing the .json files
EXCEL_FILE = r"C:\Users\ITandon2\Metadata-Enrichment\Well_Headers_and_Completion_Reports 2017.xlsx"       # excel file with the lookup table
EXCEL_SHEET = 0                        # sheet name or index
WELL_NAME_COL = "Well Name"
CONFIDENTIAL_COL = "CONFIDENTIAL"

MATCHED_JSON_FOLDER = "matched_json"   # output folder for matched jsons
PAIRS_LOG_FILE = "pdf_json_pairs.txt"  # output pairing log

RENAME_JSON_TO_PDF_NAME = True         # copy json as <pdf_name>.json
FUZZY_MATCH_THRESHOLD = 0.85           # 0-1, lower = more lenient fallback matching
# --------------------------------------------------------------------------


def normalize(text):
    """Lower-case, drop a trailing .pdf/.json, collapse punctuation/spacing."""
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\.(pdf|json)$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_facility_name(obj, target="facility name"):
    """
    Recursively search a parsed JSON object (dict/list) for a key whose
    name matches `target` once punctuation/case/underscores are stripped
    (so "facility name", "Facility_Name", "facilityName" all match).
    Returns the first value found, or None.
    """
    target_norm = re.sub(r"[^a-z0-9]", "", target.lower())

    def _search(node):
        if isinstance(node, dict):
            for key, value in node.items():
                key_norm = re.sub(r"[^a-z0-9]", "", str(key).lower())
                if key_norm == target_norm and value not in (None, ""):
                    return value
            for value in node.values():
                found = _search(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _search(item)
                if found is not None:
                    return found
        return None

    return _search(obj)


def load_pdf_to_well_map(excel_path, sheet, well_col, conf_col):
    """Returns dict: normalized(CONFIDENTIAL value) -> original Well Name."""
    df = pd.read_excel(excel_path, sheet_name=sheet)
    if well_col not in df.columns or conf_col not in df.columns:
        raise ValueError(
            f"Excel file must contain columns '{well_col}' and '{conf_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    mapping = {}
    for _, row in df.iterrows():
        confidential = row[conf_col]
        well_name = row[well_col]
        if pd.isna(confidential) or pd.isna(well_name):
            continue
        mapping[normalize(confidential)] = str(well_name).strip()
    return mapping


def build_json_facility_map(json_folder):
    """
    Recursively scans `json_folder` (and all sub-folders) for .json files.

    Returns:
        facility_map: dict normalized(facility name) -> list of
                       (json_path, original_facility_name)
        problems:     list of (json_filename, reason) for files that
                       couldn't be parsed or had no facility-name key
    """
    facility_map = {}
    problems = []

    json_paths = []
    for root, _dirs, files in os.walk(json_folder):
        for fname in files:
            if fname.lower().endswith(".json"):
                json_paths.append(os.path.join(root, fname))

    for fpath in sorted(json_paths):
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            problems.append((fname, f"could not parse JSON: {e}"))
            continue

        facility = find_facility_name(data)
        if not facility:
            problems.append((fname, "no 'facility name' key found"))
            continue

        norm = normalize(facility)
        facility_map.setdefault(norm, []).append((fpath, str(facility).strip()))

    return facility_map, problems


def best_fuzzy_match(target_norm, candidates_norm):
    """Returns (best_candidate_key, score) or (None, 0) if nothing clears the threshold."""
    best, best_score = None, 0.0
    for cand in candidates_norm:
        score = difflib.SequenceMatcher(None, target_norm, cand).ratio()
        if score > best_score:
            best, best_score = cand, score
    if best_score >= FUZZY_MATCH_THRESHOLD:
        return best, best_score
    return None, 0.0


def main():
    os.makedirs(MATCHED_JSON_FOLDER, exist_ok=True)

    print("Reading Excel lookup table...")
    pdf_to_well = load_pdf_to_well_map(EXCEL_FILE, EXCEL_SHEET, WELL_NAME_COL, CONFIDENTIAL_COL)
    print(f"  -> {len(pdf_to_well)} CONFIDENTIAL/Well Name rows loaded.\n")

    print("Scanning JSON files for facility names...")
    facility_map, json_problems = build_json_facility_map(JSON_FOLDER)
    found_count = sum(len(v) for v in facility_map.values())
    print(f"  -> {found_count} json file(s) with a facility name found.")
    if json_problems:
        print(f"  -> {len(json_problems)} json file(s) skipped (see log for details).")
    print()

    pdf_files = sorted(f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf"))
    print(f"Found {len(pdf_files)} PDF file(s) in '{PDF_FOLDER}'.\n")

    matched_lines = []
    unmatched_lines = []

    for pdf_name in pdf_files:
        pdf_norm = normalize(pdf_name)
        well_name = pdf_to_well.get(pdf_norm)

        if well_name is None:
            unmatched_lines.append(f"{pdf_name}  ->  NO EXCEL ROW FOUND (no CONFIDENTIAL match)")
            continue

        well_norm = normalize(well_name)
        json_candidates = facility_map.get(well_norm)
        match_score = 1.0

        if not json_candidates:
            best_key, score = best_fuzzy_match(well_norm, list(facility_map.keys()))
            if best_key is not None:
                json_candidates = facility_map[best_key]
                match_score = score

        if not json_candidates:
            unmatched_lines.append(
                f"{pdf_name}  ->  Well Name '{well_name}' has NO matching JSON facility name"
            )
            continue

        # If several jsons share the same facility name, copy all of them,
        # numbering duplicates so nothing gets overwritten.
        pdf_base = os.path.splitext(pdf_name)[0]
        for idx, (json_path, original_facility) in enumerate(json_candidates):
            json_name = os.path.basename(json_path)
            if RENAME_JSON_TO_PDF_NAME:
                new_name = f"{pdf_base}.json" if idx == 0 else f"{pdf_base}_{idx + 1}.json"
            else:
                new_name = json_name

            shutil.copy2(json_path, os.path.join(MATCHED_JSON_FOLDER, new_name))

            score_note = "" if match_score >= 0.999 else f"  (fuzzy match, score={match_score:.2f})"
            matched_lines.append(
                f"{pdf_name}  ->  {json_name}   [Well Name: {well_name} | Facility Name: {original_facility}]{score_note}"
            )

    with open(PAIRS_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("MATCHED PDF -> JSON PAIRS\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(matched_lines) + "\n" if matched_lines else "(no matches found)\n")

        f.write("\n\nUNMATCHED PDFs\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(unmatched_lines) + "\n" if unmatched_lines else "(none - all pdfs matched)\n")

        if json_problems:
            f.write("\n\nJSON FILES SKIPPED (no facility name found / parse error)\n")
            f.write("=" * 60 + "\n")
            f.write("\n".join(f"{name}: {reason}" for name, reason in json_problems) + "\n")

    print(f"Done. {len(matched_lines)} match(es), {len(unmatched_lines)} unmatched PDF(s).")
    print(f"Matched JSON files copied to: {MATCHED_JSON_FOLDER}")
    print(f"Pairing log written to: {PAIRS_LOG_FILE}")


if __name__ == "__main__":
    main()