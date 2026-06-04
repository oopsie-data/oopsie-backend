"""Terminal viewer for oopsiedata_format_v1 HDF5 files.

Usage:
    python -m oopsie_backend.inspect_hdf5 <path/to/episode.h5>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


# ── ANSI colours ────────────────────────────────────────────────────────────
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + _RESET


# ── helpers ──────────────────────────────────────────────────────────────────


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode(value.item())
        return _decode(value.flat[0]) if value.size else ""
    return str(value)


def _is_text(ds: h5py.Dataset) -> bool:
    return h5py.check_string_dtype(ds.dtype) is not None or ds.dtype.kind in ("S", "U", "O")


def _dataset_summary(ds: h5py.Dataset) -> str:
    # Null-dataspace datasets (created with h5py.Empty) have shape == None
    if ds.shape is None:
        return _c("Empty", _DIM)
    if ds.shape == ():
        if _is_text(ds):
            val = _decode(ds[()])
            return _c(repr(val), _GREEN)
        return _c(f"scalar {ds.dtype}", _DIM)
    shape_str = " × ".join(str(d) for d in ds.shape)
    return _c(f"({shape_str})  {ds.dtype}", _YELLOW)


def _format_json_attr(text: str, indent_prefix: str) -> str:
    """Pretty-print a JSON string with continuation lines indented."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return _c(repr(text[:120]) + ("…" if len(text) > 120 else ""), _GREEN)
    lines = json.dumps(obj, indent=2).splitlines()
    first = _c(lines[0], _GREEN)
    rest = ("\n" + indent_prefix).join(_c(l, _GREEN) for l in lines[1:])
    return first + ("\n" + indent_prefix + rest if rest else "")


def _attr_repr(value: Any, *, key: str = "", indent_prefix: str = "") -> str:
    if isinstance(value, (bytes, np.ndarray)) or isinstance(value, str):
        text = _decode(value) if not isinstance(value, str) else value
        if text.startswith("{") or text.startswith("["):
            return _format_json_attr(text, indent_prefix)
        return _c(repr(text), _GREEN)
    if isinstance(value, (np.floating, float)):
        return _c(str(float(value)), _YELLOW)
    return _c(repr(value), _GREEN)


# ── tree printer ─────────────────────────────────────────────────────────────


def _print_node(
    name: h5py.HLObject,
    obj: h5py.HLObject,
    prefix: str,
    is_last: bool,
) -> None:
    connector = "└── " if is_last else "├── "
    child_prefix = prefix + ("    " if is_last else "│   ")

    if isinstance(obj, h5py.Group):
        label = _c(name + "/", _CYAN, _BOLD)
        n_children = len(obj)
        print(f"{prefix}{connector}{label}  {_c(f'[{n_children} items]', _DIM)}")
    else:
        # Dataset (may be h5py.Empty when the file stores one)
        try:
            summary = _dataset_summary(obj)
        except Exception:
            summary = _c("(unreadable)", _DIM)
        label = _c(name, _BOLD)
        print(f"{prefix}{connector}{label}  {summary}")

        # For scalar text datasets print the value inline
        if not isinstance(obj, h5py.Empty) and obj.shape == () and _is_text(obj):
            pass  # already shown in summary

    # Attributes
    if hasattr(obj, "attrs") and obj.attrs:
        attrs = list(obj.attrs.items())
        for k, v in attrs:
            key_str = f"{child_prefix}{_c('@ ' + k, _DIM)} = "
            indent = " " * len(f"{child_prefix}@ {k} = ")
            print(key_str + _attr_repr(v, key=k, indent_prefix=indent))


def _walk(group: h5py.Group, prefix: str = "") -> None:
    items = list(group.items())
    for i, (name, obj) in enumerate(items):
        is_last = i == len(items) - 1
        _print_node(name, obj, prefix, is_last)
        child_prefix = prefix + ("    " if is_last else "│   ")
        if isinstance(obj, h5py.Group):
            _walk(obj, child_prefix)


def inspect(path: Path) -> None:
    with h5py.File(path, "r") as f:
        # Header
        schema = _decode(f.attrs.get("schema", ""))
        print()
        print(_c(str(path), _BOLD))
        if schema:
            print(_c(f"  schema: {schema}", _DIM))
        print()

        # Root attributes
        root_attrs = list(f.attrs.items())
        if root_attrs:
            print(_c("Root attributes", _CYAN, _BOLD))
            for k, v in root_attrs:
                key_str = f"  {_c(k, _BOLD)} = "
                indent = " " * (2 + len(k) + 3)
                print(key_str + _attr_repr(v, key=k, indent_prefix=indent))
            print()

        # Tree
        print(_c("Dataset tree", _CYAN, _BOLD))
        _walk(f)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an oopsiedata HDF5 file.")
    parser.add_argument("path", nargs="+", help="Path(s) to .h5 / .hdf5 file(s)")
    args = parser.parse_args()

    ok = True
    for p in args.path:
        path = Path(p)
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            ok = False
            continue
        try:
            inspect(path)
        except Exception as exc:
            print(f"error reading {path}: {exc}", file=sys.stderr)
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
