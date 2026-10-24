"""Eval split columns of the episode index.

``eval_split`` holds the held-out split of the chosen proposal (``id``,
``od_task_id_scene``, ``od_task_od_scene`` or ``excluded``); null means the
episode is not held out.
"""

from __future__ import annotations

import polars as pl

from oopsie_backend.index.store import KEY, episode_key

EVAL_SCHEMA = {KEY: pl.String, "eval_split": pl.String, "eval_proposal": pl.String}
EVAL_COLUMNS = [c for c in EVAL_SCHEMA if c != KEY]


def eval_split_frame(proposal: dict) -> pl.DataFrame:
    rows = [
        {KEY: episode_key(e["dataset_repo"], e["repo_path"]), "eval_split": split, "eval_proposal": proposal["proposal_id"]}
        for split, episodes in proposal["episodes"].items()
        for e in episodes
    ]
    return pl.DataFrame(rows, schema=EVAL_SCHEMA)
