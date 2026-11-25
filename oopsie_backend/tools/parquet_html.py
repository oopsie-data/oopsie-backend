"""Render a Parquet file as a standalone, searchable HTML table.

Preserves nested values in a row inspector and includes column types. Exports
the first 1,000 rows by default; use --max-rows 0 to include the entire file.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

TEMPLATE = Path(__file__).with_name("parquet_viewer.html")


def _json_value(value):
    """Keep Arrow values readable and preserve integers beyond JavaScript's precision."""
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return "hex:" + value.hex()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1:
        return str(value)
    return value


def render_parquet(
    source: Path, out: Path | None = None, *, max_rows: int = 1000, columns: list[str] | None = None
) -> Path:
    """Write an offline HTML viewer without modifying the source Parquet file."""
    source = Path(source).expanduser().resolve(strict=True)
    out = Path(out).expanduser().resolve() if out is not None else source.with_suffix(".html")
    if out == source or out.suffix.lower() != ".html":
        raise ValueError("Output must be an .html file different from the source")
    if max_rows < 0:
        raise ValueError("max_rows must be nonnegative; 0 includes all rows")

    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    names = schema.names if columns is None else list(dict.fromkeys(columns))
    unknown = sorted(set(names) - set(schema.names))
    if unknown:
        raise ValueError(f"Unknown columns: {', '.join(unknown)}")
    if not names:
        raise ValueError("Select at least one column")

    rows = []
    batch_size = min(max_rows or 4096, 4096)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=names):
        if max_rows:
            batch = batch.slice(0, max_rows - len(rows))
        rows.extend([_json_value(row[name]) for name in names] for row in batch.to_pylist())
        if max_rows and len(rows) >= max_rows:
            break

    payload = {
        "source": str(source),
        "name": source.name,
        "total_rows": parquet.metadata.num_rows,
        "total_columns": len(schema),
        "columns": [{"name": name, "type": str(schema.field(name).type)} for name in names],
        "rows": rows,
    }
    # Script elements parse their closing tag even inside JSON strings.
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False, default=str)
    for char, escape in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e")):
        encoded = encoded.replace(char, escape)
    document = TEMPLATE.read_text(encoding="utf-8").replace("__PARQUET_PAYLOAD__", encoded)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Parquet file to view")
    parser.add_argument("--out", type=Path, help="HTML output; defaults to the source path with .html extension")
    parser.add_argument("--max-rows", type=int, default=1000, help="First N rows to embed (default: 1000); 0 embeds all rows")
    parser.add_argument("--columns", nargs="+", help="Only export these columns; default: all columns")
    args = parser.parse_args(argv)
    try:
        out = render_parquet(args.source, args.out, max_rows=args.max_rows, columns=args.columns)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"HTML viewer written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
