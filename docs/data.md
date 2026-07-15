# Data folders & file formats

These directories hold **data, not code**. Most are git-ignored (large or
generated). Doc IDs are derived from filenames: `datalab-output-PR2610.pdf.json`
→ `PR2610`.

## Folders

| Path | Description | Tracked? |
|---|---|---|
| `datalab-output/` | Datalab extraction JSON, one per PDF | ignored |
| `matched_json/` | OSDU ground-truth record per document | ignored |
| `enriched_ground_truth/` | Human-verified ground truth (only confirmed attributes) | tracked |
| `new_checking_truth/` | Working ground-truth drafts | — |
| `pdfs/` | Source PDFs (served by the web app) | ignored |
| `chroma_store/` | Persistent Chroma vector store | ignored |
| `output_table/` | Evaluation summaries, gold sets, charts, reports | — |
| `jina-v3/` | Optional local Jina v3 model snapshot | ignored |
| `index_cache/` | Cached built indexes | ignored |

## Key files

| File | Description |
|---|---|
| `queries.txt` | Evaluation queries grouped by label (see below) |
| `useful_chunks.jsonl` | Chunk dump from a `build` run |
| `all_schema_keys.json`, `common_non_null_schema_keys.json` | Schema key inventories used by query generation |
| `Well_Headers_and_Completion_Reports 2017.xlsx` | Source spreadsheet used by `rag.data_prep.linking` |

## `queries.txt` format

Plain text grouped by a label line ending in `:`, followed by the question
phrasing(s). Labels ending in `2` are `[facility name]`-phrasing variants.

```
facility name:

What is the well or the wellbore name?

lat+long:

What is the geographic location(latitude and longitude) of this well based on its WGS84 coordinates?
```

Query labels cover: facility name, lat+long, openv ID, spud date, country,
operator, basin, field, markers, completion date, vertical measurements.

## Ground-truth formats

### OSDU matched (`matched_json/PR<id>.json`)
Full structured OSDU-style record used for standard evaluation and MTEB retrieval
metrics.

### Enriched (`enriched_ground_truth/PR<id>.json`)
A flat record under a `data` key holding only the attributes a reviewer verified,
e.g.:

```json
{
  "data": {
    "FacilityName": "Warea-1",
    "SpatialLocation.Wgs84Coordinates": "-39.2221806,174.7822389",
    "WellboreSituation": "Onshore",
    "FacilityEvents.SpudDate": "2000-11-05",
    "CountryName": "New Zealand",
    "GeoContexts.BasinID": "TARANAKI BASIN",
    "FieldName": "Taranaki",
    "CurrentOperator": "DISCOVERY GEO(AUSTRALIA) CORPORATION",
    "FacilityEvents.CompletionDate": "2000-12-10"
  }
}
```

Enriched runs query only the attributes present for each document and read ground
truth straight from this flat record (no OSDU schema traversal). MTEB retrieval
ranking metrics are not computed for enriched runs — accuracy is purely
end-to-end.

## Output layout (`output_table/`)

| Path | Produced by |
|---|---|
| `batch_runner_summary.json` | `rag/inference/batch_runner.py` |
| `batch/`, `batch_summary.json` | `rag.evaluation.batch` |
| `retrieval_goldset.json` | `rag.scripts.build_retrieval_goldset` |
| `retrieval_metrics.json` | `rag.scripts.evaluate_retrieval` |
| `matryoshka/` | `rag.scripts.run_matryoshka_study` / `analyze_matryoshka_study` |
| `tuning_best_10files.json` | `rag.scripts.aggregate_tuning_results` |
| `benchmarking/`, `benchmarking_enriched/` | `rag.benchmarking.*` |

## Preparing data

- **Extraction JSON** comes from Datalab (external step) and lands in `datalab-output/`.
- **Linking** PDFs to ground-truth JSON: edit the `CONFIGURATION` block in
  `rag/data_prep/linking.py`, install `pandas openpyxl`, then run
  `python -m rag.data_prep.linking`.
- **Local embedding model**: `python -m rag.data_prep.download`.
