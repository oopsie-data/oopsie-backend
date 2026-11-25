"""Checks for Parquet previews, nested values and safe standalone HTML output."""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from oopsie_backend.tools.parquet_html import render_parquet


def payload(path):
    document = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="parquet-data" type="application/json">(.*?)</script>', document, re.S)
    return json.loads(match.group(1))


class ParquetHTMLTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "episodes.parquet"
        self.table = pa.table({
            "episode_key": ["first", "second", "third"],
            "steps": [30, 2, None],
            "annotations": [[{"success": 1.0, "tags": ["ok"]}], [], None],
            "timestamp": [datetime(2026, 11, 8, tzinfo=timezone.utc), None, None],
        })
        pq.write_table(self.table, self.source)

    def test_preview_preserves_types_and_nested_values_without_changing_source(self):
        before = self.source.read_bytes()
        out = render_parquet(self.source, max_rows=2)
        data = payload(out)
        self.assertEqual(out, self.source.with_suffix(".html"))
        self.assertEqual(data["total_rows"], 3)
        self.assertEqual(data["total_columns"], 4)
        self.assertEqual(len(data["rows"]), 2)
        self.assertEqual(data["rows"][0][2], [{"success": 1.0, "tags": ["ok"]}])
        self.assertEqual(data["rows"][0][3], "2026-11-08T00:00:00+00:00")
        self.assertEqual(data["columns"][2]["type"], str(pq.read_schema(self.source).field("annotations").type))
        self.assertEqual(self.source.read_bytes(), before)

    def test_all_rows_column_order_and_custom_destination(self):
        destination = Path(self.tmp.name) / "reports" / "index.html"
        data = payload(render_parquet(self.source, destination, max_rows=0, columns=["steps", "episode_key"]))
        self.assertEqual([c["name"] for c in data["columns"]], ["steps", "episode_key"])
        self.assertEqual(data["rows"], [[30, "first"], [2, "second"], [None, "third"]])

    def test_empty_file_still_exports_schema(self):
        pq.write_table(self.table.slice(0, 0), self.source)
        data = payload(render_parquet(self.source))
        self.assertEqual(data["rows"], [])
        self.assertEqual(data["total_rows"], 0)
        self.assertEqual(len(data["columns"]), 4)

    def test_script_delimiters_and_special_arrow_values(self):
        unsafe = '</script><script>alert("x")</script> & <b>hello</b>'
        table = pa.table({
            "text": [unsafe],
            "large_int": [2**60 + 1],
            "nan": [float("nan")],
            "infinity": [float("inf")],
            "binary": [b'\x00\xff'],
            "decimal": [Decimal("1.23")],
        })
        pq.write_table(table, self.source)
        out = render_parquet(self.source)
        document = out.read_text(encoding="utf-8")
        self.assertNotIn(unsafe, document)
        self.assertNotIn('<script>alert(', document)
        self.assertEqual(payload(out)["rows"][0], [unsafe, str(2**60 + 1), "nan", "inf", "hex:00ff", "1.23"])
        self.assertNotIn('src="http', document)

    def test_invalid_options_do_not_overwrite_source(self):
        before = self.source.read_bytes()
        for kwargs in ({"out": self.source}, {"max_rows": -1}, {"columns": ["missing"]}, {"columns": []}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                render_parquet(self.source, **kwargs)
        self.assertEqual(self.source.read_bytes(), before)

    def test_default_limit(self):
        pq.write_table(pa.table({"row": range(1100)}), self.source)
        data = payload(render_parquet(self.source))
        self.assertEqual(data["total_rows"], 1100)
        self.assertEqual(len(data["rows"]), 1000)
        self.assertEqual(data["rows"][-1], [999])


if __name__ == "__main__":
    unittest.main()
