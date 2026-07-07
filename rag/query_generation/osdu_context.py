"""OSDU attribute knowledge used to ground query generation.

The queries in ``queries.txt`` target attributes of the **OSDU Well / Wellbore
master-data model** (Open Subsurface Data Universe). To make the LLM generate a
retrieval query that actually matches how NZ petroleum well *reports* phrase an
attribute, we give it a short, curated "interpretation" of the OSDU field plus a
few well-report phrasings we already know are common.

This registry is intentionally lightweight: it seeds the prompt for the
attributes we know, and for anything unknown the prompt asks the model to infer
the OSDU field from the attribute name. Add entries here as new attributes are
introduced - nothing else needs to change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class OsduAttribute:
    """Curated hint for one OSDU attribute."""

    osdu_field: str                       # e.g. "FacilityEvents[].EffectiveDateTime"
    meaning: str                          # one-line human description
    report_terms: List[str] = field(default_factory=list)  # phrasings seen in reports


# Normalised attribute label -> curated OSDU hint. Keys are lower-cased and have
# any trailing generation marker / numeric variant suffix stripped (see
# ``normalize_attribute``), so "Marker Names", "marker names2" and "marker names*"
# all resolve to the same hint.
OSDU_ATTRIBUTES: Dict[str, OsduAttribute] = {
    "facility name": OsduAttribute(
        "FacilityName / WellName",
        "The name/identifier of the well or facility.",
        ["well name", "wellbore name", "borehole name", "facility name"],
    ),
    "lat+long": OsduAttribute(
        "SpatialLocation.Wgs84Coordinates",
        "Surface geographic location of the well (latitude/longitude).",
        ["surface location", "wellhead coordinates", "latitude and longitude",
         "easting northing", "degrees minutes seconds", "NZTM", "NZGD2000", "WGS84"],
    ),
    "openv id": OsduAttribute(
        "OperatingEnvironmentID",
        "Whether the well is onshore or offshore.",
        ["onshore", "offshore", "operating environment", "location setting"],
    ),
    "facilityevent spud date": OsduAttribute(
        "FacilityEvents[].EffectiveDateTime (event type = spud)",
        "The date drilling of the well commenced (spud).",
        ["spud date", "spudded", "kick off date", "commenced drilling",
         "start of drilling", "spud in"],
    ),
    "country name": OsduAttribute(
        "CountryName",
        "The country in which the well is located.",
        ["country", "nation"],
    ),
    "default vertical measurement id": OsduAttribute(
        "DefaultVerticalMeasurementID",
        "The vertical measurement datum / reference elevation for depths.",
        ["datum", "depth reference", "reference elevation", "KB", "kelly bushing",
         "RT", "rotary table", "GL", "ground level", "MSL", "derrick floor"],
    ),
    "current operator id": OsduAttribute(
        "CurrentOperatorID / CurrentOperator",
        "The company currently operating the well.",
        ["operator", "operated by", "well operator", "operating company", "licensee"],
    ),
    "basin name": OsduAttribute(
        "GeoContexts[].BasinID",
        "The sedimentary basin the well is located in.",
        ["basin", "sedimentary basin", "petroleum basin"],
    ),
    "field name": OsduAttribute(
        "FieldName",
        "The named petroleum field the well belongs to (not the basin or area).",
        ["field", "oil field", "gas field", "field/area", "prospect"],
    ),
    "marker names": OsduAttribute(
        "Wellbore markers / formation tops",
        "Named stratigraphic markers / formation tops penetrated by the well.",
        ["formation tops", "stratigraphic tops", "markers", "formation markers",
         "lithostratigraphy", "stratigraphy", "formations", "members",
         "geological tops", "tops"],
    ),
    "completion date": OsduAttribute(
        "FacilityEvents[] (completion / abandonment)",
        "When the well was completed, suspended, or plugged and abandoned.",
        ["completion date", "completed", "rig released", "reached total depth",
         "suspended", "plugged and abandoned", "P&A", "end of well"],
    ),
    "verticalmeasurementid kb": OsduAttribute(
        "VerticalMeasurements[] (reference = Kelly Bushing)",
        "Vertical measurement referenced to Kelly Bushing (KB) plus TVD/trajectory.",
        ["kelly bushing", "KB", "KB elevation", "above kelly bushing", "AHBKB",
         "true vertical depth", "TVD", "trajectory", "deviation"],
    ),
    "vertical measurement total depth": OsduAttribute(
        "VerticalMeasurements[] (type = total depth)",
        "The total depth of the well as a measured depth (MD).",
        ["total depth", "TD", "measured depth", "MD", "final depth", "reached TD",
         "trajectory", "deviation survey"],
    ),
}


# Trailing generation marker ("*") and/or numeric variant suffix ("2"), so
# "marker names*", "marker names2" and "marker names*2" all normalise back to
# "marker names". Anchored at the end only, so cores like "lat+long" survive.
_SUFFIX_RE = re.compile(r"[\*\d]+$")


def normalize_attribute(attribute: str) -> str:
    """Lower-case an attribute label and strip any ``*`` / numeric variant suffix."""
    key = (attribute or "").strip().lower()
    stripped = _SUFFIX_RE.sub("", key).strip()
    return stripped or key


def lookup(attribute: str) -> Optional[OsduAttribute]:
    """Return the curated OSDU hint for an attribute, or None when unknown."""
    return OSDU_ATTRIBUTES.get(normalize_attribute(attribute))


# Raw OSDU field names (as they appear in a Well/Wellbore schema or record) mapped
# to the friendly attribute labels used in queries.txt. This lets a field pulled
# straight out of an uploaded schema (e.g. "FacilityName") reuse the curated hint
# AND resolve to the same ground-truth extractor as the hand-written queries.
# Keys are lower-cased. Fields with no entry keep their raw name (the prompt then
# asks the LLM to infer their meaning).
OSDU_FIELD_TO_LABEL: Dict[str, str] = {
    "facilityname": "facility name",
    "wellname": "facility name",
    "operatingenvironmentid": "openv ID",
    "wellboresituation": "openv ID",
    "spatiallocation.wgs84coordinates": "lat+long",
    "countryname": "country name",
    "defaultverticalmeasurementid": "default vertical measurement ID",
    "currentoperatorid": "current operator ID",
    "currentoperator": "current operator ID",
    "fieldname": "field name",
    "markernames": "marker names",
}


def friendly_label(field_name: str) -> str:
    """Return the friendly attribute label for a raw OSDU field, or the field itself."""
    return OSDU_FIELD_TO_LABEL.get((field_name or "").strip().lower(), field_name)
