"""Per-lab requirement checks (alpha/beta tiers).

A lab is "alpha" if any of its dataset repos has a real creation date before the
configured cutoff (``requirements.alpha_cutoff_date`` in configs/stats_config.yaml);
otherwise it's "beta". A lab *passes* its tier when its episodes, summed over all
of its datasets and operators, clear every enabled requirement clause for that tier.
``contributors_supported`` is how many times the totals clear the bar, i.e. how
many contributors the lab's submission covers when each needs the full bar.

Thresholds, the cutoff date, and which clauses are enabled all live in
configs/stats_config.yaml.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from oopsie_backend.common.repos import (
    ARCHIVE_DATASETS,
    EXTERNAL_DATASETS,
    INFRA_DATASETS,
    canonical_dataset_repo,
)
from oopsie_backend.common.stats_config import CONFIG
from oopsie_backend.stats.episode_stats import (
    _include_in_public_stats,
    _primary_sf_annotation,
)

# Status values attached to each lab (and, via datasets, each inventory row):
#   "pass"          - the lab's totals clear every enabled bar for its tier
#   "fail"          - the lab has episodes, but its totals don't clear the bar
#   "no_data"       - the lab has no episodes yet
#   "not_evaluated" - that lab's tier has no enabled requirement clauses right now


def _is_placeholder_operator(name: str | None, placeholder_names: frozenset[str]) -> bool:
    norm = (name or "").strip()
    return (not norm) or (norm.lower() in placeholder_names)


def _classify_lab_type(dataset_repos: set[str], created_at: dict[str, str | None], cutoff: str) -> str:
    dates = [created_at.get(d) for d in dataset_repos if created_at.get(d)]
    is_alpha = any(d[:10] < cutoff for d in dates)
    return "alpha" if is_alpha else "beta"


def _tier_status_if_disabled(lab_type: str, req: dict) -> str | None:
    """Return 'not_evaluated' if the lab's tier has nothing enabled to check, else None.

    task_diversity/per_task_cap aren't implemented yet (see _contributors_supported), so
    their `enabled` flags are intentionally ignored here - only `volume` can make
    beta "evaluated" today, regardless of what those two stub flags say.
    """
    if lab_type == "alpha":
        return None if req["alpha"]["enabled"] else "not_evaluated"
    return None if req["beta"]["volume"]["enabled"] else "not_evaluated"


def _times(have: float, need: float) -> int | None:
    """How many times ``have`` covers ``need``; None when ``need`` is not positive."""
    return int(have // need) if need > 0 else None


def _contributors_supported(stats: dict, lab_type: str, req: dict) -> int:
    """How many contributors the lab's totals cover when each needs the full bar of the tier."""
    if lab_type == "alpha":
        fits = [_times(stats["qualifying"], req["alpha"]["min_qualifying_episodes"])]
    else:
        # task_diversity / per_task_cap: not yet implemented, never limit the count.
        v = req["beta"]["volume"]
        fits = [
            _times(stats["qualifying"], v["min_qualifying_episodes"]),
            _times(stats["failures"], v["min_failures"]),
            _times(stats["duration_seconds"] / 3600.0, v["min_duration_hours"]),
        ]
    fits = [n for n in fits if n is not None]
    return min(fits) if fits else int(stats["qualifying"] > 0)


def _repo_short_name(dataset: str) -> str:
    return dataset.split("/", 1)[-1] if "/" in dataset else dataset


def evaluate_lab_requirements(
    records: list[dict],
    repo: str,
    dataset_created_at: dict[str, str | None],
    all_dataset_repos: list[str] | None = None,
) -> dict[str, dict]:
    """Per-lab alpha/beta requirement evaluation.

    ``dataset_created_at`` should map canonical dataset repo id -> an ISO date/
    datetime string (or None); pass the *effective* dates from the dataset
    inventory (HF repo date, falling back to earliest episode) for the most
    complete picture.

    ``all_dataset_repos``, if given, seeds a ``"no_data"`` entry (lab id guessed
    from the repo name) for every known dataset repo that had no episode records
    at all, so e.g. a lab that registered but hasn't submitted yet still gets a
    correctly-typed (alpha/beta) entry instead of being silently absent.

    Returns ``{lab_id: {...}}`` - see module docstring for the ``status`` values.
    """
    req = CONFIG["requirements"]
    placeholder_names = CONFIG["placeholder_operator_names"]
    cutoff = req["alpha_cutoff_date"]
    never_a_lab = ARCHIVE_DATASETS | EXTERNAL_DATASETS | INFRA_DATASETS

    lab_datasets: defaultdict[str, set[str]] = defaultdict(set)
    per_lab: defaultdict[str, dict] = defaultdict(
        lambda: {"episodes": 0, "qualifying": 0, "failures": 0, "duration_seconds": 0.0}
    )
    placeholder_counts: defaultdict[str, Counter] = defaultdict(Counter)

    for d in all_dataset_repos or []:
        canon = canonical_dataset_repo(d)
        if canon in never_a_lab:
            continue
        lab_datasets[_repo_short_name(canon)].add(canon)

    for r in records:
        if not _include_in_public_stats(r, repo):
            continue
        lab = r["lab_id"]
        dataset = canonical_dataset_repo(r.get("dataset_repo") or repo)
        lab_datasets[lab].add(dataset)

        op_norm = (r.get("operator_name") or "").strip()
        if _is_placeholder_operator(op_norm, placeholder_names):
            placeholder_counts[lab][op_norm or "(blank)"] += 1

        bucket = per_lab[lab]
        bucket["episodes"] += 1
        primary = _primary_sf_annotation(r["annotations"])
        if primary is not None:
            bucket["qualifying"] += 1
            if primary["success"] < 0.5:
                bucket["failures"] += 1
            if r["duration_seconds"]:
                bucket["duration_seconds"] += r["duration_seconds"]

    result: dict[str, dict] = {}
    for lab in sorted(set(lab_datasets) | set(per_lab)):
        lab_type = _classify_lab_type(lab_datasets.get(lab, set()), dataset_created_at, cutoff)
        disabled_status = _tier_status_if_disabled(lab_type, req)
        stats = per_lab.get(lab)
        supported = None
        if disabled_status is None and stats is not None:
            supported = _contributors_supported(stats, lab_type, req)

        if disabled_status is not None:
            status = disabled_status
        elif stats is None:
            status = "no_data"
        elif supported:
            status = "pass"
        else:
            status = "fail"

        result[lab] = {
            "lab_type": lab_type,
            "status": status,
            "requirement_stats": (
                {
                    "qualifying_episodes": stats["qualifying"],
                    "failures": stats["failures"],
                    "duration_hours": round(stats["duration_seconds"] / 3600.0, 3),
                    "contributors_supported": supported,
                }
                if stats is not None
                else None
            ),
            "placeholder_operator_episodes": sum(placeholder_counts[lab].values()),
            "placeholder_operator_names": sorted(placeholder_counts[lab]),
        }
    return result
