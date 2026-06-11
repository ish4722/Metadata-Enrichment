"""
STEP 2 — FIELD SCHEMA REGISTRY
===============================
Defines every field we want to extract from the OCR text.
Each field has:
  - name        : field name used in extraction
  - type        : expected Python type as string
  - required    : whether absence should be flagged
  - source_hints: what text patterns to look for
  - example     : sample valid value
  - validation  : callable that returns (bool, reason_str)

Import WELL_SCHEMA anywhere you need the field definitions.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import re


# ---------------------------------------------------------------------------
# Field descriptor
# ---------------------------------------------------------------------------

@dataclass
class FieldDef:
    name: str                          # field name used in extraction
    type: str                          # "string" | "float" | "date" | "coordinate" | "list"
    required: bool
    source_hints: list[str]            # phrases the LLM should scan for
    example: Any                       # sample value shown in prompt
    validation: Callable[[Any], tuple[bool, str]]  # (ok, reason)
    osdu_path: str = ""                # dot-notation path in final JSON
    notes: str = ""                    # extra context for the LLM prompt


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _nonempty_str(v):
    if v is None:
        return False, "value is null"
    if not isinstance(v, str) or not v.strip():
        return False, "must be a non-empty string"
    return True, ""

def _any_val(v):
    return True, ""   # accept anything including null

def _positive_float(v):
    if v is None:
        return True, ""   # allowed to be null
    try:
        f = float(v)
        if f < 0:
            return False, f"depth/elevation must be >= 0, got {f}"
        return True, ""
    except (TypeError, ValueError):
        return False, f"expected a number, got {v!r}"

def _latitude(v):
    if v is None:
        return True, ""
    try:
        f = float(v)
        if not (-90 <= f <= 90):
            return False, f"latitude must be -90..90, got {f}"
        return True, ""
    except (TypeError, ValueError):
        return False, f"expected a float, got {v!r}"

def _longitude(v):
    if v is None:
        return True, ""
    try:
        f = float(v)
        if not (-180 <= f <= 180):
            return False, f"longitude must be -180..180, got {f}"
        return True, ""
    except (TypeError, ValueError):
        return False, f"expected a float, got {v!r}"

def _iso_date(v):
    if v is None:
        return True, ""
    if isinstance(v, str):
        if re.match(r"\d{4}-\d{2}-\d{2}", v):
            return True, ""
        return False, f"date must be YYYY-MM-DD format, got {v!r}"
    return False, f"expected string date, got {type(v).__name__}"

def _operating_env(v):
    if v is None:
        return True, ""
    allowed = {"offshore", "onshore", "subsea", "deepwater", "shallow water"}
    if isinstance(v, str) and v.lower() in allowed:
        return True, ""
    return False, f"expected one of {allowed}, got {v!r}"

def _well_name(v):
    if v is None:
        return False, "well name is required"
    if isinstance(v, str) and re.match(r"^[A-Za-z0-9\-_/ ]{1,64}$", v.strip()):
        return True, ""
    return False, f"well name has unexpected characters or length: {v!r}"


# ---------------------------------------------------------------------------
# Full schema
# ---------------------------------------------------------------------------

WELL_SCHEMA: list[FieldDef] = [

    # ── Core identity ────────────────────────────────────────────────────────
    FieldDef(
        name="FacilityName",
        type="string",
        required=True,
        source_hints=["well name", "facility name", "borehole name", "hole name", "well id"],
        example="Pateke-4H",
        validation=_well_name,
        notes="The official designation of the well. Often appears near the top of a report."
    ),
    FieldDef(
        name="CountryName",
        type="string",
        required=True,
        source_hints=["country", "nation", "state/country", "location country"],
        example="New Zealand",
        validation=_nonempty_str,
    ),
    FieldDef(
        name="FieldName",
        type="string",
        required=False,
        source_hints=["field name", "oil field", "gas field", "field:", "producing field"],
        example="Pateke",
        validation=_any_val,
    ),
    FieldDef(
        name="BlockName",
        type="string",
        required=False,
        source_hints=["block", "block name", "license block", "permit block", "acreage block"],
        example="38158",
        validation=_any_val,
    ),
    FieldDef(
        name="ProspectName",
        type="string",
        required=False,
        source_hints=["prospect", "prospect name", "exploration prospect"],
        example="Alpha Prospect",
        validation=_any_val,
    ),
    FieldDef(
        name="RegionName",
        type="string",
        required=False,
        source_hints=["region", "basin region", "geographic region", "area"],
        example="Taranaki Basin",
        validation=_any_val,
    ),

    # ── Spatial ─────────────────────────────────────────────────────────────
    FieldDef(
        name="Latitude_WGS84",
        type="float",
        required=True,
        source_hints=["latitude", "lat", "y coordinate", "northing (wgs)", "decimal degrees north"],
        example=-39.3757361,
        validation=_latitude,
        notes="Decimal degrees. Negative = south. Expect 4-7 decimal places."
    ),
    FieldDef(
        name="Longitude_WGS84",
        type="float",
        required=True,
        source_hints=["longitude", "long", "lon", "x coordinate", "easting (wgs)", "decimal degrees east"],
        example=173.1959694,
        validation=_longitude,
        notes="Decimal degrees. Negative = west."
    ),

    # ── Operating environment ────────────────────────────────────────────────
    FieldDef(
        name="OperatingEnvironment",
        type="string",
        required=False,
        source_hints=["offshore", "onshore", "operating environment", "location type", "marine", "subsea", "water depth"],
        example="Offshore",
        validation=_operating_env,
        notes="Infer from context if not explicit — if water depth is present, it is likely Offshore."
    ),

    # ── Facility / well type ─────────────────────────────────────────────────
    FieldDef(
        name="FacilityType",
        type="string",
        required=False,
        source_hints=["well type", "facility type", "borehole type", "type of well"],
        example="Well",
        validation=_any_val,
    ),
    FieldDef(
        name="WellStatus",
        type="string",
        required=False,
        source_hints=["well status", "status", "current status", "well condition", "plugged", "abandoned", "producing"],
        example="Plugged and Abandoned",
        validation=_any_val,
    ),

    # ── Operator ─────────────────────────────────────────────────────────────
    FieldDef(
        name="OperatorName",
        type="string",
        required=False,
        source_hints=["operator", "operator name", "current operator", "company", "licensee"],
        example="Australian Worldwide Exploration",
        validation=_any_val,
        notes="Extract the raw company name — URN resolution happens later."
    ),
    FieldDef(
        name="InitialOperatorName",
        type="string",
        required=False,
        source_hints=["initial operator", "original operator", "drilled by", "drilling contractor"],
        example="Australian Worldwide Exploration",
        validation=_any_val,
    ),

    # ── Dates / events ───────────────────────────────────────────────────────
    FieldDef(
        name="SpudDate",
        type="date",
        required=False,
        source_hints=["spud date", "spud", "drilling start", "drill start", "date spudded", "commencement date"],
        example="2014-02-09",
        validation=_iso_date,
        notes="Return in YYYY-MM-DD format. Convert 'Feb 9, 2014' or '09/02/2014' accordingly."
    ),
    FieldDef(
        name="CompletionDate",
        type="date",
        required=False,
        source_hints=["completion date", "completed", "total depth date", "td date", "well completion"],
        example="2014-04-15",
        validation=_iso_date,
    ),
    FieldDef(
        name="AbandonmentDate",
        type="date",
        required=False,
        source_hints=["abandonment date", "abandoned", "p&a date", "plugged and abandoned", "well abandoned"],
        example="2014-05-01",
        validation=_iso_date,
    ),

    # ── Vertical measurements ────────────────────────────────────────────────
    FieldDef(
        name="RT_Elevation_m",
        type="float",
        required=False,
        source_hints=["rotary table", "rt elevation", "rt", "kelly bushing", "kb elevation", "drill floor elevation"],
        example=26.0,
        validation=_positive_float,
        notes="In metres. If given in feet, convert to metres (1 ft = 0.3048 m)."
    ),
    FieldDef(
        name="WellHeadElevation_m",
        type="float",
        required=False,
        source_hints=["well head elevation", "wellhead elevation", "wellhead height", "surface elevation"],
        example=26.0,
        validation=_positive_float,
    ),
    FieldDef(
        name="WaterDepth_m",
        type="float",
        required=False,
        source_hints=["water depth", "sea depth", "depth of water", "water column", "ocean depth"],
        example=124.0,
        validation=_positive_float,
        notes="Only present for offshore wells. In metres."
    ),
    FieldDef(
        name="TotalDepth_m",
        type="float",
        required=False,
        source_hints=["total depth", "td", "final depth", "total drilled depth", "bottom hole depth"],
        example=3500.0,
        validation=_positive_float,
    ),

    # ── Identifiers ──────────────────────────────────────────────────────────
    FieldDef(
        name="UniqueWellIdentifier",
        type="string",
        required=False,
        source_hints=["uwi", "unique well identifier", "unique well id", "api number", "well number"],
        example="100000790909",
        validation=_any_val,
    ),
    FieldDef(
        name="LicenseNumber",
        type="string",
        required=False,
        source_hints=["license", "licence", "permit", "exploration permit", "permit number"],
        example="PEP38158",
        validation=_any_val,
    ),

    # ── Data provenance ──────────────────────────────────────────────────────
    FieldDef(
        name="DataSource",
        type="string",
        required=False,
        source_hints=["source", "data source", "data provider", "provided by", "supplied by"],
        example="New Zealand Petroleum and Minerals",
        validation=_any_val,
    ),
    FieldDef(
        name="ExistenceKind",
        type="string",
        required=False,
        source_hints=["existence kind", "actual", "planned", "proposed", "conceptual"],
        example="Actual",
        validation=_any_val,
    ),
]


# ---------------------------------------------------------------------------
# Helper: build the schema block injected into the LLM prompt
# ---------------------------------------------------------------------------

def schema_to_prompt_block(schema: list[FieldDef]) -> str:
    """
    Serialises the schema into a compact JSON array
    suitable for injection into the extraction prompt.
    """
    import json
    fields = []
    for f in schema:
        fields.append({
            "field_name": f.name,
            "type": f.type,
            "required": f.required,
            "look_for": f.source_hints,
            "example_value": f.example,
            "notes": f.notes or "",
        })
    return json.dumps(fields, indent=2)


if __name__ == "__main__":
    print(f"Schema contains {len(WELL_SCHEMA)} field definitions.\n")
    for f in WELL_SCHEMA:
        req = "REQUIRED" if f.required else "optional"
        print(f"  [{req:8s}] {f.name:35s}  → {f.osdu_path}")
