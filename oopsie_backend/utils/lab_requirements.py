"""Per-individual lab requirement checks (alpha/beta tiers).

A lab is "alpha" if any of its dataset repos has a real creation date before the
configured cutoff (``requirements.alpha_cutoff_date`` in configs/stats_config.yaml);
otherwise it's "beta". A lab *passes* its tier if even one individual (operator)
within it independently clears every enabled requirement clause for that tier -
requirements are never summed across people within a lab.

Thresholds, the cutoff date, and which clauses are enabled all live in
configs/stats_config.yaml.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from oopsie_backend.utils.episode_stats import (
    _include_in_public_stats,
    _primary_sf_annotation,
    canonical_dataset_repo,
)
from oopsie_backend.utils.stats_config import CONFIG

# Status values attached to each lab (and, via datasets, each inventory row):
#   "pass"          - at least one individual clears every enabled bar for their tier
#   "fail"          - real episode data exists, but no individual clears the bar
#   "no_data"       - no qualifying (non-placeholder) individual data at all yet
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

    task_diversity/per_task_cap aren't implemented yet (see _meets_requirement), so
    their `enabled` flags are intentionally ignored here - only `volume` can make
    beta "evaluated" today, regardless of what those two stub flags say.
    """
    if lab_type == "alpha":
        return None if req["alpha"]["enabled"] else "not_evaluated"
    return None if req["beta"]["volume"]["enabled"] else "not_evaluated"


def _meets_requirement(stats: dict, lab_type: str, req: dict) -> bool:
    if lab_type == "alpha":
        a = req["alpha"]
        if not a["enabled"]:
            return False
        return stats["qualifying"] >= a["min_qualifying_episodes_per_individual"]

    beta = req["beta"]
    ok = True
    if beta["volume"]["enabled"]:
        v = beta["volume"]
        hours = stats["duration_seconds"] / 3600.0
        ok = ok and (
            stats["qualifying"] >= v["min_qualifying_episodes_per_individual"]
            and stats["failures"] >= v["min_failures_per_individual"]
            and hours >= v["min_duration_hours_per_individual"]
        )
    # task_diversity / per_task_cap: not yet implemented, never block a pass.
    return ok


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
    never_a_lab = CONFIG["stats_excluded_datasets"] | CONFIG["public_excluded_datasets"] | CONFIG["non_lab_infra_datasets"]

    lab_datasets: defaultdict[str, set[str]] = defaultdict(set)
    per_individual: defaultdict[tuple[str, str], dict] = defaultdict(
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

        op_raw = r.get("operator_name")
        op_norm = (op_raw or "").strip()
        if _is_placeholder_operator(op_norm, placeholder_names):
            placeholder_counts[lab][op_norm or "(blank)"] += 1
            continue

        bucket = per_individual[(lab, op_norm)]
        bucket["episodes"] += 1
        primary = _primary_sf_annotation(r["annotations"])
        if primary is not None:
            bucket["qualifying"] += 1
            if primary["success"] < 0.5:
                bucket["failures"] += 1
            if r["duration_seconds"]:
                bucket["duration_seconds"] += r["duration_seconds"]

    labs = set(lab_datasets) | {lab for (lab, _op) in per_individual}
    result: dict[str, dict] = {}
    for lab in sorted(labs):
        lab_type = _classify_lab_type(lab_datasets.get(lab, set()), dataset_created_at, cutoff)
        disabled_status = _tier_status_if_disabled(lab_type, req)

        individuals = {op: s for (l, op), s in per_individual.items() if l == lab}
        passing = [op for op, s in individuals.items() if _meets_requirement(s, lab_type, req)]

        if disabled_status is not None:
            status = disabled_status
        elif passing:
            status = "pass"
        elif individuals:
            status = "fail"
        else:
            status = "no_data"

        qualifying_individual = passing[0] if passing else None
        # For the tooltip: show the passer, or else whoever got closest (ranked by
        # qualifying-episode count, the shared first bar for both tiers).
        best_individual = None
        best_stats = None
        if qualifying_individual is not None:
            best_individual = qualifying_individual
            best_stats = individuals[qualifying_individual]
        elif individuals:
            best_individual = max(individuals, key=lambda op: individuals[op]["qualifying"])
            best_stats = individuals[best_individual]

        result[lab] = {
            "lab_type": lab_type,
            "status": status,
            "qualifying_individual": qualifying_individual,
            "individuals_considered": len(individuals),
            "best_individual": best_individual,
            "best_individual_stats": (
                {
                    "qualifying_episodes": best_stats["qualifying"],
                    "failures": best_stats["failures"],
                    "duration_hours": round(best_stats["duration_seconds"] / 3600.0, 3),
                }
                if best_stats is not None
                else None
            ),
            "placeholder_operator_episodes": sum(placeholder_counts[lab].values()),
            "placeholder_operator_names": sorted(placeholder_counts[lab]),
        }
    return result
