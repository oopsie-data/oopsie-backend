"""Extract per-episode metadata from Oopsie HDF5 files and aggregate dashboard stats."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np

from oopsie_backend.utils.stats_config import CONFIG

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
REPORTS_DIR = _BACKEND_ROOT / "reports"
DEFAULT_PUBLIC_STATS_JSON = REPORTS_DIR / "public_stats.json"
DEFAULT_PUBLIC_STATS_HTML = REPORTS_DIR / "public_stats.html"
DEFAULT_DOCS_STATS_JSON = Path(
    os.environ.get(
        "OOPSIE_STATS_OUT",
        _REPO_ROOT / "oopsie-docs/docs/assets/data/stats.json",
    )
)
DEFAULT_INTERNAL_STATS_JSON = REPORTS_DIR / "internal_stats.json"
# Back-compat alias for scripts that import DEFAULT_STATS_JSON.
DEFAULT_STATS_JSON = DEFAULT_DOCS_STATS_JSON

PEOPLE_SCHEMA_VERSION = 3
INTERNAL_SCHEMA_VERSION = 9
UNKNOWN_PERSON = "unknown"

SCHEMA_VERSION = 5
TOP_TASKS_LIMIT = 20
UNKNOWN_LAB = "unknown_lab"
UNKNOWN_ROBOT = "unknown_robot"
UNKNOWN_POLICY = "unknown_policy"
UNKNOWN_CATEGORY = "unknown"
UNKNOWN_SEVERITY = "unknown"
DAY_MISSING_TIMESTAMP = "unrecorded"  # legacy label; missing timestamps are omitted from timelines

# Some submissions (e.g. Harvard_EML) write a relative/monotonic clock value into the
# "timestamp" attribute instead of a wall-clock Unix timestamp, which _iso() would
# otherwise convert to a bogus date near 1970-01-01. Anything older than this is treated
# as implausible and stamped with the placeholder below instead.
_MIN_PLAUSIBLE_TIMESTAMP = datetime(2000, 1, 1, tzinfo=timezone.utc)
_IMPLAUSIBLE_TIMESTAMP_FALLBACK = "2026-09-01T00:00:00Z"

# The "Over time" charts start at this day; episodes dated earlier (real historical
# submissions from pre-existing datasets, e.g. Berkeley_RAIL) are folded into it so the
# timeline doesn't stretch back before the project's actual public launch.
TIMELINE_FLOOR_DAY = "2026-05-01"

# The lists below live in configs/stats_config.yaml, not here - edit that file,
# not this one. (Re-run compute_local_stats/compute_hf_stats for edits to take
# effect; these are read once at import time.)

# Never included in public OR internal stats (archives / frozen snapshots).
STATS_EXCLUDED_DATASETS: frozenset[str] = CONFIG["stats_excluded_datasets"]

# Bulk-import / legacy HF repos omitted from the public stats.json entirely.
# (Still appear in internal stats unless also listed in STATS_EXCLUDED_DATASETS.)
PUBLIC_EXCLUDED_DATASETS: frozenset[str] = CONFIG["public_excluded_datasets"]
# Placeholder / unset lab_ids omitted from the public website stats.
PUBLIC_EXCLUDED_LABS: frozenset[str] = CONFIG["public_excluded_labs"]
# Back-compat alias used in existing docs and call sites.
PER_LAB_EXCLUDED_DATASETS = PUBLIC_EXCLUDED_DATASETS

# HF org repos that hold site assets (preview images/video, metadata) rather than any
# lab's episode submissions. Excluded from the internal dataset inventory so it only
# lists real labs.
NON_LAB_INFRA_DATASETS: frozenset[str] = CONFIG["non_lab_infra_datasets"]

HF_ORG = "OopsieData-Submissions"

CENTRAL_TZ = ZoneInfo("America/Chicago")


def now_timestamps() -> dict[str, str]:
    """Return current time as UTC and US Central (CDT/CST) ISO strings."""
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    now_central = now_utc.astimezone(CENTRAL_TZ)
    return {
        "utc": now_utc.isoformat().replace("+00:00", "Z"),
        "cdt": now_central.isoformat(),
    }


_ACTION_KEYS = (
    "joint_position",
    "joint_velocity",
    "cartesian_position",
    "cartesian_velocity",
    "gripper_position",
    "gripper_velocity",
    "gripper_binary",
    "base_position",
    "base_velocity",
)


def _s(v):
    """Decode an HDF5 attr value to a python str (or None)."""
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        return _s(v.item()) if v.size else None
    if isinstance(v, (np.bytes_, np.str_)):
        return _s(v.tolist())
    return str(v)


def _attr(g, key):
    return g.attrs[key] if key in g.attrs else None


def _profile(raw):
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return d if isinstance(d, dict) else {}


def _str_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x is not None]
    return []


def _n_timesteps(group):
    if group is None:
        return None
    for k in _ACTION_KEYS:
        ds = group.get(k)
        if isinstance(ds, h5py.Dataset) and ds.shape and ds.shape[0] > 0:
            return int(ds.shape[0])
    return None


def _iso(ts):
    if ts is None:
        return None
    try:
        f = float(np.asarray(ts).item())
    except (TypeError, ValueError):
        return _s(ts)
    try:
        dt = datetime.fromtimestamp(f, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    if dt < _MIN_PLAUSIBLE_TIMESTAMP:
        return _IMPLAUSIBLE_TIMESTAMP_FALLBACK
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _extract_record_from_h5(f: h5py.File, repo_path: str) -> dict:
    profile = _profile(_s(_attr(f, "robot_profile")))

    ann_group = f.get("episode_annotations")
    annotations = []
    if isinstance(ann_group, h5py.Group):
        for name, g in ann_group.items():
            if not isinstance(g, h5py.Group):
                continue
            tax_raw = _s(_attr(g, "taxonomy"))
            tax = {}
            if tax_raw:
                try:
                    decoded = json.loads(tax_raw)
                    if isinstance(decoded, dict):
                        tax = decoded
                except (json.JSONDecodeError, TypeError):
                    pass
            success = _attr(g, "success")
            desc = _s(_attr(g, "failure_description")) or ""
            annotations.append(
                {
                    "name": str(name),
                    "timestamp": _s(_attr(g, "timestamp")),
                    "success": float(np.asarray(success).item())
                    if success is not None
                    else None,
                    "failure_category": _s(tax.get("failure_category"))
                    or UNKNOWN_CATEGORY,
                    "severity": _s(tax.get("severity")) or UNKNOWN_SEVERITY,
                    "has_failure_description": bool(desc.strip()),
                    "failure_description": desc.strip(),
                }
            )

    actions = f.get("actions") if isinstance(f.get("actions"), h5py.Group) else None
    obs = f.get("observations") if isinstance(f.get("observations"), h5py.Group) else None
    video_paths = (
        obs.get("video_paths")
        if isinstance(obs, h5py.Group) and isinstance(obs.get("video_paths"), h5py.Group)
        else None
    )

    try:
        control_freq = float(profile.get("control_freq")) if profile.get("control_freq") else None
    except (TypeError, ValueError):
        control_freq = None
    n = _n_timesteps(actions)
    duration = (n / control_freq) if (n and control_freq) else None

    return {
        "repo_path": repo_path,
        "episode_id": _s(_attr(f, "episode_id")) or "",
        "lab_id": _s(_attr(f, "lab_id")) or UNKNOWN_LAB,
        "operator_name": _s(_attr(f, "operator_name")),
        "language_instruction": _s(_attr(f, "language_instruction")),
        "timestamp": _iso(_attr(f, "timestamp")),
        "robot_name": _s(profile.get("robot_name")) or UNKNOWN_ROBOT,
        "policy_name": _s(profile.get("policy_name")) or UNKNOWN_POLICY,
        "action_space": _str_list(profile.get("action_space")),
        "cameras_present": sorted(video_paths.keys()) if video_paths else [],
        "duration_seconds": duration,
        "annotations": annotations,
    }


def extract_record(source: bytes | Path, repo_path: str) -> dict:
    """Pull the per-episode metadata we need for aggregation."""
    if isinstance(source, Path):
        with h5py.File(source, "r") as f:
            return _extract_record_from_h5(f, repo_path)
    with h5py.File(BytesIO(source), "r") as f:
        return _extract_record_from_h5(f, repo_path)


def _primary_annotation(anns: list[dict]) -> dict | None:
    if not anns:
        return None
    return max(anns, key=lambda a: (1 if a.get("timestamp") else 0, a.get("timestamp") or ""))


def _primary_sf_annotation(anns: list[dict]) -> dict | None:
    """Latest annotation that has a success/failure label (``success`` attr)."""
    with_sf = [a for a in anns if a.get("success") is not None]
    return _primary_annotation(with_sf) if with_sf else None


def _is_complete_annotation(ann: dict) -> bool:
    """True when an annotation record counts as a finished annotation.

    Used for annotator people-stats and language-diversity corpora. Public
    outcome totals (success / failure / annotated) use ``_primary_sf_annotation``
    instead — S/F label only.
    """
    if ann.get("success") is None:
        return False
    if ann["success"] >= 0.5:
        return True
    if not ann.get("has_failure_description"):
        return False
    if ann.get("failure_category") == UNKNOWN_CATEGORY:
        return False
    if ann.get("severity") == UNKNOWN_SEVERITY:
        return False
    return True


def _primary_complete_annotation(anns: list[dict]) -> dict | None:
    """Latest complete annotation by timestamp, if any."""
    complete = [a for a in anns if _is_complete_annotation(a)]
    return _primary_annotation(complete) if complete else None


def canonical_dataset_repo(dataset: str) -> str:
    """Normalize any dataset label to ``namespace/repo`` for comparisons."""
    if dataset.startswith("local:"):
        path_parts = Path(dataset[6:]).parts
        for i, part in enumerate(path_parts):
            if part == HF_ORG and i + 1 < len(path_parts):
                return f"{HF_ORG}/{path_parts[i + 1]}"
        return dataset

    parts = dataset.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == HF_ORG:
        return f"{parts[0]}/{parts[1]}"
    if parts and parts[0] != HF_ORG:
        return f"{HF_ORG}/{parts[0]}"
    return dataset


def resolve_local_episode_path(
    data_dir: Path, rel: Path
) -> tuple[str, str] | None:
    """Map a local ``.h5`` path to ``(dataset_repo, repo_relative_path)``."""
    parts = rel.parts
    if data_dir.name == HF_ORG:
        if len(parts) < 2:
            return None
        return f"{HF_ORG}/{parts[0]}", "/".join(parts[1:])
    if len(parts) < 3:
        return None
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def _include_in_any_stats(record: dict, repo: str) -> bool:
    """True when an episode may appear in public or internal stats."""
    raw = record.get("dataset_repo") or repo
    return canonical_dataset_repo(raw) not in STATS_EXCLUDED_DATASETS


def _include_in_public_stats(record: dict, repo: str) -> bool:
    """True when an episode should appear in public stats.json / the website."""
    if not _include_in_any_stats(record, repo):
        return False
    lab = record.get("lab_id") or UNKNOWN_LAB
    if lab in PUBLIC_EXCLUDED_LABS:
        return False
    raw = record.get("dataset_repo") or repo
    return canonical_dataset_repo(raw) not in PUBLIC_EXCLUDED_DATASETS


def _normalize_person_name(name: str | None) -> str:
    if not name or not str(name).strip():
        return UNKNOWN_PERSON
    return str(name).strip()


def write_stats_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def write_public_artifacts(
    stats: dict,
    *,
    backend_json_path: Path | None = None,
    docs_json_path: Path | None = None,
    html_path: Path | None = None,
    write_html: bool = True,
) -> None:
    """Write public stats JSON (backend + docs site) and optional HTML snapshot."""
    backend_json = backend_json_path or DEFAULT_PUBLIC_STATS_JSON
    docs_json = docs_json_path or DEFAULT_DOCS_STATS_JSON
    html_out = html_path or DEFAULT_PUBLIC_STATS_HTML
    write_stats_json(backend_json, stats)
    write_stats_json(docs_json, stats)
    if write_html:
        from oopsie_backend.utils.public_report import render_public_report

        render_public_report(stats, html_out)


def public_dataset_repos(dataset_ids: list[str]) -> list[str]:
    """HF repo ids that appear in public stats (non-excluded only)."""
    return sorted(
        d
        for d in dataset_ids
        if d not in PUBLIC_EXCLUDED_DATASETS and d not in STATS_EXCLUDED_DATASETS
    )


def aggregate_people(
    records: list[dict], repo: str, *, include_excluded: bool = False
) -> dict:
    """Aggregate operator and annotator counts per lab.

    By default uses the public corpus filter (``PUBLIC_EXCLUDED_DATASETS`` omitted).
    Pass ``include_excluded=True`` to count all repos.

    Annotator counts include only **complete** annotation records: ``success`` is
    set; failures additionally require a non-empty description and non-unknown
    category and severity.
    """
    per_lab_operators: defaultdict[str, Counter] = defaultdict(Counter)
    per_lab_annotators: defaultdict[str, Counter] = defaultdict(Counter)
    all_operators: set[str] = set()
    all_annotators: set[str] = set()
    operator_episodes = 0
    annotation_count = 0

    for r in records:
        if not _include_in_any_stats(r, repo):
            continue
        if not include_excluded and not _include_in_public_stats(r, repo):
            continue
        lab = r["lab_id"]
        operator = _normalize_person_name(r.get("operator_name"))
        per_lab_operators[lab][operator] += 1
        operator_episodes += 1
        if operator != UNKNOWN_PERSON:
            all_operators.add(operator)

        for ann in r.get("annotations") or []:
            if not _is_complete_annotation(ann):
                continue
            annotator = _normalize_person_name(ann.get("name"))
            per_lab_annotators[lab][annotator] += 1
            annotation_count += 1
            if annotator != UNKNOWN_PERSON:
                all_annotators.add(annotator)

    per_lab = {}
    for lab in sorted(set(per_lab_operators) | set(per_lab_annotators)):
        per_lab[lab] = {
            "operators": dict(per_lab_operators[lab].most_common()),
            "annotators": dict(per_lab_annotators[lab].most_common()),
            "operator_episodes": int(sum(per_lab_operators[lab].values())),
            "annotation_count": int(sum(per_lab_annotators[lab].values())),
        }

    return {
        "schema_version": PEOPLE_SCHEMA_VERSION,
        "generated_at": now_timestamps(),
        "dataset_repo": repo,
        "totals": {
            "operator_episodes": operator_episodes,
            "annotations": annotation_count,
            "labs": len(per_lab),
            "distinct_operators": len(all_operators),
            "distinct_annotators": len(all_annotators),
        },
        "per_lab": per_lab,
    }


def build_dataset_inventory(
    per_dataset: dict[str, dict],
    all_dataset_repos: list[str],
    dataset_created_at: dict[str, str | None] | None = None,
) -> dict[str, dict]:
    """Merge computed per-dataset episode/hour stats with the complete list of known
    HF repos, zero-filling any lab that hasn't submitted episodes yet.

    ``created_at`` prefers the dataset's real HF repo creation date (from
    ``dataset_created_at``, e.g. fetched live from the Hub); falling back to the
    timestamp of its earliest episode when the Hub date isn't available, and to
    ``None`` (no signal at all) otherwise. ``created_at_source`` records which one.

    Archives (``STATS_EXCLUDED_DATASETS``) and non-lab infra repos
    (``NON_LAB_INFRA_DATASETS``) are always left out.
    """
    created_by_canonical: dict[str, str | None] = {
        canonical_dataset_repo(repo_id): created
        for repo_id, created in (dataset_created_at or {}).items()
    }

    known = {canonical_dataset_repo(d) for d in all_dataset_repos}
    known |= set(per_dataset)  # never drop a dataset that actually has episodes
    known -= STATS_EXCLUDED_DATASETS
    known -= NON_LAB_INFRA_DATASETS

    inventory = {}
    for dataset in sorted(known):
        existing = per_dataset.get(dataset, {})
        hf_created = created_by_canonical.get(dataset)
        if hf_created:
            created_at, created_at_source = hf_created, "hf_repo_created"
        elif existing.get("first_submission"):
            created_at, created_at_source = existing["first_submission"], "first_episode"
        else:
            created_at, created_at_source = None, None
        inventory[dataset] = {
            "episodes": int(existing.get("episodes") or 0),
            "duration_hours": float(existing.get("duration_hours") or 0.0),
            "bulk_import": dataset in PUBLIC_EXCLUDED_DATASETS,
            "created_at": created_at,
            "created_at_source": created_at_source,
        }
    return inventory


def _repo_short_name_guess(dataset: str) -> str:
    return dataset.split("/", 1)[-1] if "/" in dataset else dataset


def _attach_requirement_status(
    inventory: dict[str, dict], per_dataset: dict[str, dict], lab_requirements: dict[str, dict]
) -> None:
    """Mutate ``inventory`` in place, tagging each dataset with its primary lab's
    alpha/beta requirement status (see :mod:`oopsie_backend.utils.lab_requirements`).

    The "primary" lab for a dataset is whichever lab contributed the most episodes
    to it (almost always the only one); datasets with no episodes yet fall back to
    guessing the lab id from the repo name, which ``evaluate_lab_requirements``
    also does when seeding its ``no_data`` entries, so the two stay in sync.
    """
    for dataset, entry in inventory.items():
        lab_ids = (per_dataset.get(dataset) or {}).get("lab_ids") or {}
        primary_lab = max(lab_ids, key=lambda lab: lab_ids[lab]) if lab_ids else _repo_short_name_guess(dataset)
        info = lab_requirements.get(primary_lab)
        entry["lab_id"] = primary_lab
        entry["lab_type"] = info["lab_type"] if info else None
        entry["requirement_status"] = info["status"] if info else "no_data"
        entry["qualifying_individual"] = info["qualifying_individual"] if info else None
        entry["best_individual"] = info["best_individual"] if info else None
        entry["best_individual_stats"] = info["best_individual_stats"] if info else None
        entry["individuals_considered"] = info["individuals_considered"] if info else 0
        entry["placeholder_operator_episodes"] = info["placeholder_operator_episodes"] if info else 0
        entry["placeholder_operator_names"] = info["placeholder_operator_names"] if info else []


def build_internal_report(
    records: list[dict],
    repo: str,
    *,
    all_dataset_repos: list[str] | None = None,
    dataset_created_at: dict[str, str | None] | None = None,
) -> dict:
    """Full stats + people breakdown for local/internal use (includes excluded repos).

    When ``all_dataset_repos`` is given, the report also includes a
    ``dataset_inventory`` covering every known dataset repo, including ones with
    zero episodes.
    """
    from oopsie_backend.utils.language_diversity import aggregate_language_diversity

    people = aggregate_people(records, repo, include_excluded=False)
    stats = aggregate(records, repo, include_excluded=True)

    # Same bulk/archive exclusion as the dataset inventory's non-bulk totals (see
    # _non_bulk_totals in internal_report.py) - kept in lock-step so the "All
    # datasets" summary cards and this action-space breakdown sum to the same
    # episode count.
    non_bulk_excluded = STATS_EXCLUDED_DATASETS | PUBLIC_EXCLUDED_DATASETS
    action_spaces_non_bulk: Counter = Counter()
    for r in records:
        dataset = canonical_dataset_repo(r.get("dataset_repo") or repo)
        if dataset in non_bulk_excluded:
            continue
        for a in r["action_space"]:
            action_spaces_non_bulk[a] += 1

    report = {
        "schema_version": INTERNAL_SCHEMA_VERSION,
        "generated_at": now_timestamps(),
        "dataset_repo": repo,
        "excluded_from_public_datasets": sorted(PUBLIC_EXCLUDED_DATASETS),
        "excluded_from_all_stats_datasets": sorted(STATS_EXCLUDED_DATASETS),
        "stats": stats,
        "action_spaces_non_bulk": dict(action_spaces_non_bulk.most_common()),
        "language_diversity": aggregate_language_diversity(
            records, repo, include_excluded=False, scope_label="public_corpus"
        ),
        "people": {
            "totals": people["totals"],
            "per_lab": people["per_lab"],
        },
    }
    if all_dataset_repos is not None:
        inventory = build_dataset_inventory(stats["per_dataset"], all_dataset_repos, dataset_created_at)
        effective_created_at = {repo_id: info["created_at"] for repo_id, info in inventory.items()}

        from oopsie_backend.utils.lab_requirements import evaluate_lab_requirements

        lab_requirements = evaluate_lab_requirements(
            records, repo, effective_created_at, all_dataset_repos
        )
        _attach_requirement_status(inventory, stats["per_dataset"], lab_requirements)
        report["dataset_inventory"] = inventory
        report["lab_requirements"] = lab_requirements
    return report


def write_internal_artifacts(
    records: list[dict],
    repo: str,
    json_path: Path | None,
    *,
    write_html: bool = True,
    html_path: Path | None = None,
    all_dataset_repos: list[str] | None = None,
    dataset_created_at: dict[str, str | None] | None = None,
) -> None:
    """Write internal_stats.json and optional HTML report."""
    if json_path is None:
        return
    report = build_internal_report(
        records,
        repo,
        all_dataset_repos=all_dataset_repos,
        dataset_created_at=dataset_created_at,
    )
    write_stats_json(json_path, report)
    if write_html:
        from oopsie_backend.utils.internal_report import render_internal_report

        render_internal_report(report, html_path or json_path.with_suffix(".html"))


def write_people_artifacts(
    records: list[dict],
    repo: str,
    json_path: Path | None,
    *,
    write_html: bool = True,
    html_path: Path | None = None,
) -> None:
    """Deprecated alias for :func:`write_internal_artifacts`."""
    write_internal_artifacts(
        records,
        repo,
        json_path,
        write_html=write_html,
        html_path=html_path,
    )


def aggregate(records: list[dict], repo: str, *, include_excluded: bool = False) -> dict:
    per_lab_eps = Counter()
    per_lab_succ = Counter()
    per_lab_fail = Counter()
    per_lab_dur = defaultdict(float)
    per_lab_robots = defaultdict(set)
    per_lab_policies = defaultdict(set)
    per_lab_tasks = defaultdict(set)
    per_lab_first = {}
    per_lab_last = {}

    per_robot_eps = Counter()
    per_robot_dur = defaultdict(float)
    per_robot_labs = defaultdict(set)

    per_policy_eps = Counter()
    per_task_eps = Counter()
    failure_categories = Counter()
    severity = Counter()
    action_spaces = Counter()
    camera_setups = Counter()
    timeline = Counter()
    timeline_per_lab: Counter = Counter()

    per_lab_desc = Counter()
    per_dataset_eps = Counter()
    per_dataset_succ = Counter()
    per_dataset_fail = Counter()
    per_dataset_dur = defaultdict(float)
    per_dataset_first = {}
    per_dataset_last = {}
    per_dataset_desc = Counter()
    per_dataset_lab_eps: defaultdict[str, Counter] = defaultdict(Counter)
    total_dur = 0.0
    annotated = successes = failures = failure_descriptions = 0
    episode_count = 0

    for r in records:
        if not _include_in_any_stats(r, repo):
            continue
        if not include_excluded and not _include_in_public_stats(r, repo):
            continue
        episode_count += 1
        lab = r["lab_id"]
        dataset = canonical_dataset_repo(r.get("dataset_repo") or repo)
        per_dataset_eps[dataset] += 1
        per_dataset_lab_eps[dataset][lab] += 1
        if r["duration_seconds"]:
            per_dataset_dur[dataset] += r["duration_seconds"]
        per_lab_eps[lab] += 1
        if r["duration_seconds"]:
            per_lab_dur[lab] += r["duration_seconds"]
            total_dur += r["duration_seconds"]
        per_lab_robots[lab].add(r["robot_name"])
        per_lab_policies[lab].add(r["policy_name"])
        if r["language_instruction"]:
            per_lab_tasks[lab].add(r["language_instruction"])
            per_task_eps[r["language_instruction"]] += 1

        per_robot_eps[r["robot_name"]] += 1
        per_robot_labs[r["robot_name"]].add(lab)
        if r["duration_seconds"]:
            per_robot_dur[r["robot_name"]] += r["duration_seconds"]

        per_policy_eps[r["policy_name"]] += 1

        for a in r["action_space"]:
            action_spaces[a] += 1

        if r["cameras_present"]:
            camera_setups[", ".join(r["cameras_present"])] += 1

        ts = r["timestamp"]
        if ts:
            if lab not in per_lab_first or ts < per_lab_first[lab]:
                per_lab_first[lab] = ts
            if lab not in per_lab_last or ts > per_lab_last[lab]:
                per_lab_last[lab] = ts
            if dataset not in per_dataset_first or ts < per_dataset_first[dataset]:
                per_dataset_first[dataset] = ts
            if dataset not in per_dataset_last or ts > per_dataset_last[dataset]:
                per_dataset_last[dataset] = ts
            day = max(ts[:10], TIMELINE_FLOOR_DAY)
            timeline[day] += 1
            timeline_per_lab[(day, lab)] += 1

        # Outcome totals: success/failure label only (no taxonomy required).
        primary = _primary_sf_annotation(r["annotations"])
        if primary is not None:
            annotated += 1
            if primary["success"] >= 0.5:
                successes += 1
                per_dataset_succ[dataset] += 1
                per_lab_succ[lab] += 1
            else:
                failures += 1
                per_dataset_fail[dataset] += 1
                per_lab_fail[lab] += 1
                if primary.get("has_failure_description"):
                    failure_descriptions += 1
                    per_dataset_desc[dataset] += 1
                    per_lab_desc[lab] += 1
                    category = primary.get("failure_category")
                    if category and category != UNKNOWN_CATEGORY:
                        failure_categories[category] += 1
                    sev = primary.get("severity")
                    if sev and sev != UNKNOWN_SEVERITY:
                        severity[sev] += 1

    submission_timeline = []
    cumulative = 0
    for date, count in sorted(timeline.items()):
        cumulative += count
        submission_timeline.append({"date": date, "episodes": count, "cumulative": cumulative})

    submission_timeline_per_lab = [
        {"date": date, "lab_id": lab_id, "episodes": int(count)}
        for (date, lab_id), count in sorted(timeline_per_lab.items())
    ]

    per_lab = {
        lab: {
            "episodes": int(per_lab_eps[lab]),
            "successes": int(per_lab_succ[lab]),
            "failures": int(per_lab_fail[lab]),
            "annotated_episodes": int(per_lab_succ[lab]) + int(per_lab_fail[lab]),
            "failure_descriptions": int(per_lab_desc[lab]),
            "duration_hours": round(per_lab_dur[lab] / 3600.0, 3),
            "robots": sorted(per_lab_robots[lab]),
            "policies": sorted(per_lab_policies[lab]),
            "task_count": len(per_lab_tasks[lab]),
            "first_submission": per_lab_first.get(lab),
            "last_submission": per_lab_last.get(lab),
        }
        for lab in sorted(per_lab_eps)
    }

    per_robot = {
        robot: {
            "episodes": int(per_robot_eps[robot]),
            "labs": len(per_robot_labs[robot]),
            "duration_hours": round(per_robot_dur[robot] / 3600.0, 3),
        }
        for robot in sorted(per_robot_eps)
    }

    per_policy = {p: {"episodes": int(c)} for p, c in sorted(per_policy_eps.items())}

    per_dataset = {}
    for dataset in sorted(per_dataset_eps):
        lab_ids = {lab: int(c) for lab, c in sorted(per_dataset_lab_eps[dataset].items())}
        per_dataset[dataset] = {
            "episodes": int(per_dataset_eps[dataset]),
            "successes": int(per_dataset_succ[dataset]),
            "failures": int(per_dataset_fail[dataset]),
            "annotated_episodes": int(per_dataset_succ[dataset]) + int(per_dataset_fail[dataset]),
            "failure_descriptions": int(per_dataset_desc[dataset]),
            "duration_hours": round(per_dataset_dur[dataset] / 3600.0, 3),
            "lab_ids": lab_ids,
            "bulk_import": dataset in PUBLIC_EXCLUDED_DATASETS,
            "first_submission": per_dataset_first.get(dataset),
            "last_submission": per_dataset_last.get(dataset),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_timestamps(),
        "dataset_repo": repo,
        "totals": {
            "episodes": episode_count,
            "labs": len(per_lab_eps),
            "datasets": len(per_dataset_eps),
            "robots": len(per_robot_eps),
            "policies": len(per_policy_eps),
            "tasks": len(per_task_eps),
            "duration_hours": round(total_dur / 3600.0, 3),
            "annotated_episodes": annotated,
            "successes": successes,
            "failures": failures,
            "failure_descriptions": failure_descriptions,
            "read_errors": 0,
        },
        "per_lab": per_lab,
        "per_dataset": per_dataset,
        "per_robot": per_robot,
        "per_policy": per_policy,
        "top_tasks": [
            {"instruction": t, "episodes": c}
            for t, c in per_task_eps.most_common(TOP_TASKS_LIMIT)
        ],
        "failure_categories": dict(failure_categories.most_common()),
        "severity_distribution": dict(severity.most_common()),
        "action_spaces": dict(action_spaces.most_common()),
        "camera_setups": dict(camera_setups.most_common()),
        "submission_timeline": submission_timeline,
        "submission_timeline_per_lab": submission_timeline_per_lab,
    }
    if include_excluded:
        result["excluded_from_public_datasets"] = sorted(PUBLIC_EXCLUDED_DATASETS)
        result["excluded_from_all_stats_datasets"] = sorted(STATS_EXCLUDED_DATASETS)
    return result
