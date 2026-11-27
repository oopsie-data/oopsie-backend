"""Eval split columns of the episode index.

``eval_split`` holds the held-out split of the chosen proposal (``id``,
``od_task_id_scene``, ``od_task_od_scene`` or ``excluded``); null means the
episode is not held out. Episodes of a held-out lab or task that the proposal
does not list, such as later uploads, are ``excluded``.
"""

from __future__ import annotations

import polars as pl

from oopsie_backend.index.store import KEY, episode_key
from oopsie_backend.splits.eval_split import EXCLUDED

EVAL_SCHEMA = {KEY: pl.String, "eval_split": pl.String, "eval_proposal": pl.String}
EVAL_COLUMNS = [c for c in EVAL_SCHEMA if c != KEY]


def eval_split_frame(proposal: dict, index: pl.DataFrame) -> pl.DataFrame:
    """Split of every listed episode, and ``excluded`` for unlisted ``index`` rows (``KEY``, ``lab_id``, ``task``) of held-out labs or tasks."""
    rows = [
        {KEY: episode_key(e["dataset_repo"], e["repo_path"]), "eval_split": split, "eval_proposal": proposal["proposal_id"]}
        for split, episodes in proposal["episodes"].items()
        for e in episodes
    ]
    listed = pl.DataFrame(rows, schema=EVAL_SCHEMA)
    held_out = (
        index.filter(
            pl.col("lab_id").is_in(proposal["ood_labs"]).fill_null(False)
            | pl.col("task").is_in([t["task"] for t in proposal["od_tasks"]]).fill_null(False)
        )
        .join(listed, on=KEY, how="anti")
        .select(KEY, pl.lit(EXCLUDED).alias("eval_split"), pl.lit(proposal["proposal_id"]).alias("eval_proposal"))
        .cast(EVAL_SCHEMA)
    )
    return pl.concat([listed, held_out])
