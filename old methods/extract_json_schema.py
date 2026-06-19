import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Set, Tuple

DEFAULT_INPUT_PATHS = [
    r"c:\Users\ITandon2\Downloads\master-data--Well",
    r"c:\Users\ITandon2\Downloads\master-data--Wellbore",
    r"c:\Users\ITandon2\Downloads\work-product-component--Document",
    r"c:\Users\ITandon2\Downloads\work-product-component--WellboreMarkerSet",
]
DEFAULT_OUTPUT_PATH = r"c:\Users\ITandon2\Metadata-Enrichment\all_schema_keys.json"
DEFAULT_COMMON_NON_NULL_OUTPUT_PATH = r"c:\Users\ITandon2\Metadata-Enrichment\common_non_null_schema_keys.json"


def input_label(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        return p.name
    if p.is_file():
        return p.stem
    return path


def iter_json_files(paths: Iterable[str]) -> Iterable[Tuple[Path, str]]:
    for path in paths:
        p = Path(path)
        label = input_label(path)
        if p.is_dir():
            for child in p.rglob('*.json'):
                yield child, label
        elif p.is_file() and p.suffix.lower() == '.json':
            yield p, label


def collect_schema_keys(data: Any, prefix: str = '') -> Set[str]:
    keys: Set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            full_path = f'{prefix}.{key}' if prefix else key
            keys.add(full_path)
            keys.update(collect_schema_keys(value, full_path))
    elif isinstance(data, list):
        for item in data:
            keys.update(collect_schema_keys(item, prefix))
    return keys


def collect_schema_values(data: Any, prefix: str = '', result=None):
    if result is None:
        result = {}
    if isinstance(data, dict):
        for key, value in data.items():
            full_path = f'{prefix}.{key}' if prefix else key
            result.setdefault(full_path, []).append(value)
            collect_schema_values(value, full_path, result)
    elif isinstance(data, list):
        for item in data:
            collect_schema_values(item, prefix, result)
    return result


def load_json_file(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract all JSON schema key paths from one or more folders and save them to a file.'
    )
    parser.add_argument(
        'inputs',
        nargs='*',
        default=DEFAULT_INPUT_PATHS,
        help='Optional directories or JSON file paths to scan. Uses built-in defaults when omitted.',
    )
    parser.add_argument(
        '--output-file',
        '-o',
        default=DEFAULT_OUTPUT_PATH,
        help=f'Output JSON file containing the list of unique key paths. Default: {DEFAULT_OUTPUT_PATH}',
    )
    parser.add_argument(
        '--common-non-null-output',
        default=DEFAULT_COMMON_NON_NULL_OUTPUT_PATH,
        help=f'Output JSON file for keys common to all inputs with non-null values. Default: {DEFAULT_COMMON_NON_NULL_OUTPUT_PATH}',
    )
    args = parser.parse_args()

    key_labels: dict[str, Set[str]] = {}
    key_non_null_labels: dict[str, Set[str]] = {}
    json_files = list(iter_json_files(args.inputs))

    if not json_files:
        raise SystemExit('No JSON files found in the provided input paths.')

    for json_path, label in json_files:
        try:
            data = load_json_file(json_path)
        except json.JSONDecodeError as exc:
            print(f'Skipping invalid JSON file: {json_path} ({exc})')
            continue
        except OSError as exc:
            print(f'Unable to read file: {json_path} ({exc})')
            continue

        schema_keys = collect_schema_keys(data)
        schema_values = collect_schema_values(data)
        for key_path in schema_keys:
            key_labels.setdefault(key_path, set()).add(label)
            values = schema_values.get(key_path, [])
            if any(value is not None for value in values):
                key_non_null_labels.setdefault(key_path, set()).add(label)

    all_labels = sorted({label for _, label in json_files})
    sorted_keys = sorted(key_labels)

    path_entries = []
    for key_path in sorted_keys:
        labels = key_labels[key_path]
        non_null_labels = key_non_null_labels.get(key_path, set())
        path_entries.append({
            'key_path': key_path,
            'present_in': sorted(labels),
            'common_to_all': len(labels) == len(all_labels),
            'unique_to': next(iter(labels)) if len(labels) == 1 else None,
            'non_null_in': sorted(non_null_labels),
            'non_null_in_all_labels': len(non_null_labels) == len(all_labels),
        })

    output = {
        'count': len(sorted_keys),
        'labels': all_labels,
        'key_paths': path_entries,
        'common_key_count': sum(1 for entry in path_entries if entry['common_to_all']),
        'common_non_null_key_count': sum(1 for entry in path_entries if entry['common_to_all'] and entry['non_null_in_all_labels']),
        'unique_key_count': sum(1 for entry in path_entries if entry['unique_to'] is not None),
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    common_non_null_output = Path(args.common_non_null_output)
    common_non_null_output.parent.mkdir(parents=True, exist_ok=True)
    common_non_null_entries = [
        entry['key_path']
        for entry in path_entries
        if entry['common_to_all'] and entry['non_null_in_all_labels']
    ]
    common_non_null_payload = {
        'count': len(common_non_null_entries),
        'labels': all_labels,
        'common_non_null_key_paths': common_non_null_entries,
    }
    with common_non_null_output.open('w', encoding='utf-8') as f:
        json.dump(common_non_null_payload, f, indent=2)

    print(f'Wrote {len(sorted_keys)} unique JSON key paths to: {out_path}')
    print(f'Wrote {len(common_non_null_entries)} common non-null key paths to: {common_non_null_output}')


if __name__ == '__main__':
    main()
