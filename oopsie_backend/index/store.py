"""Read and update the per-episode parquet index.

The index is one parquet file with one row per episode, keyed by
``episode_key``. ``build_episode_index`` owns the core columns; every other
pipeline step owns its own columns and writes them with ``update_columns``.
Writes hold an exclusive lock on ``<index>.lock`` and replace the file
atomically, so concurrent steps never lose each other's columns.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl

KEY = "episode_key"
DEFAULT_INDEX = Path("episode_index.parquet")


def episode_key(dataset_repo: str, repo_path: str) -> str:
    return hashlib.sha1(f"{dataset_repo}/{repo_path}".encode()).hexdigest()[:16]


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Exclusive lock for a read-modify-write of the index at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def read_index(
    path: Path = DEFAULT_INDEX, columns: list[str] | None = None, optional: dict[str, pl.DataType] | None = None
) -> pl.DataFrame:
    """The index, or ``KEY`` plus ``columns``. ``optional`` columns a step has not written yet are added as nulls."""
    optional = optional or {}
    present = set(pl.read_parquet_schema(path))
    if columns is not None:
        wanted = dict.fromkeys([KEY, *columns, *optional])
        columns = [c for c in wanted if c in present or c not in optional]
    df = pl.read_parquet(path, columns=columns)
    return df.with_columns(pl.lit(None, dtype=t).alias(c) for c, t in optional.items() if c not in df.columns)


def write_index(df: pl.DataFrame, path: Path) -> None:
    """Replace the index file atomically; call inside ``locked``."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        df.sort(KEY).write_parquet(tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def columns_fingerprint(path: Path, columns: list[str]) -> str:
    """Hash of ``columns`` (those present) over all episodes, for detecting changes."""
    if not path.is_file():
        return "absent"
    present = [c for c in columns if c in pl.read_parquet_schema(path)]
    df = read_index(path, present).sort(KEY)
    h = hashlib.sha256(",".join(df.columns).encode())
    h.update(df.hash_rows(seed=0).to_numpy().tobytes())
    return h.hexdigest()[:16]


def update_columns(
    path: Path, updates: pl.DataFrame, columns: list[str], clear_missing: bool = False
) -> pl.DataFrame:
    """Set ``columns`` from ``updates`` (which holds ``KEY``) and return the new index.

    With ``clear_missing`` the columns are reset to null for every episode not
    in ``updates``, so a step that recomputes its full output leaves no stale
    values. Keys missing from the index are an error.
    """
    updates = updates.select(KEY, *columns)
    if updates[KEY].is_duplicated().any():
        raise ValueError(f"duplicate {KEY} values in update")
    with locked(path):
        df = pl.read_parquet(path)
        unknown = updates.join(df.select(KEY), on=KEY, how="anti")
        if unknown.height:
            raise ValueError(f"{unknown.height} {KEY} value(s) not in {path}, e.g. {unknown[KEY][0]}")
        if clear_missing:
            df = df.drop([c for c in columns if c in df.columns]).join(updates, on=KEY, how="left")
        else:
            df = df.with_columns(
                pl.lit(None, dtype=updates.schema[c]).alias(c) for c in columns if c not in df.columns
            )
            df = df.update(updates, on=KEY, how="left", include_nulls=True)
        write_index(df, path)
    return df


def known_only(df: pl.DataFrame, index: Path, what: str) -> pl.DataFrame:
    """Rows of ``df`` whose key is in the index; reports the others."""
    keys = read_index(index, columns=[KEY])
    missing = df.join(keys, on=KEY, how="anti")
    if missing.height:
        print(f"  {missing.height} {what} episode(s) not in {index}; not recorded", file=sys.stderr)
    return df.join(keys, on=KEY, how="semi")
