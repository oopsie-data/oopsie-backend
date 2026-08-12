from __future__ import annotations

import json
import random
import sys
from collections import Counter

import pytest

from oopsie_backend.scripts import export_public_visualizer_subset as exporter


def make_episodes(dataset: str, count: int, *, success: float = 0.0) -> list[dict]:
    return [
        {
            "dataset_repo": f"OopsieData-Submissions/{dataset}",
            "episode_id": f"{dataset}-{index}",
            "success": success,
        }
        for index in range(count)
    ]


def test_dataset_name_supports_remote_and_generated_metadata() -> None:
    assert exporter.dataset_name({"hf_path": "Armnet/episode_000001.h5"}) == "Armnet"
    assert (
        exporter.dataset_name(
            {"dataset_repo": "OopsieData-Submissions/Berkeley_RAIL"}
        )
        == "Berkeley_RAIL"
    )
    assert (
        exporter.dataset_name({"src": "videos/CMU_Intent/run/episode__front.mp4"})
        == "CMU_Intent"
    )


def test_allocate_dataset_quotas_is_exact_bounded_and_represents_small_sets() -> None:
    episodes = (
        make_episodes("large", 100)
        + make_episodes("medium", 25)
        + make_episodes("small", 2)
    )

    quotas = exporter.allocate_dataset_quotas(
        episodes,
        target=30,
        minimum=5,
        weight_power=0.5,
    )

    assert sum(quotas.values()) == 30
    assert quotas["small"] == 2
    assert 5 <= quotas["medium"] <= 25
    assert 5 <= quotas["large"] <= 100
    assert quotas["large"] > quotas["medium"]


def test_equal_weighting_distributes_slots_evenly() -> None:
    episodes = (
        make_episodes("a", 10)
        + make_episodes("b", 10)
        + make_episodes("c", 10)
    )

    quotas = exporter.allocate_dataset_quotas(
        episodes,
        target=9,
        minimum=0,
        weight_power=0,
    )

    assert quotas == {"a": 3, "b": 3, "c": 3}


def test_allocate_dataset_quotas_rejects_an_infeasible_minimum() -> None:
    episodes = (
        make_episodes("a", 10)
        + make_episodes("b", 10)
        + make_episodes("c", 10)
    )

    with pytest.raises(ValueError, match="lower the minimum"):
        exporter.allocate_dataset_quotas(
            episodes,
            target=5,
            minimum=2,
        )


def test_sample_by_dataset_quota_is_deterministic_and_exact() -> None:
    episodes = make_episodes("a", 10) + make_episodes("b", 10)
    quotas = {"a": 3, "b": 2}

    first = exporter.sample_by_dataset_quota(episodes, quotas, random.Random(42))
    second = exporter.sample_by_dataset_quota(episodes, quotas, random.Random(42))

    assert [episode["episode_id"] for episode in first] == [
        episode["episode_id"] for episode in second
    ]
    assert Counter(exporter.dataset_name(episode) for episode in first) == {
        "a": 3,
        "b": 2,
    }


def test_main_writes_balanced_anonymized_subset(tmp_path, monkeypatch) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    videos_root = input_root / "videos"
    videos_root.mkdir(parents=True)

    videos = []
    for dataset in ("alpha", "beta", "gamma"):
        for outcome_name, success in (("failure", 0.0), ("success", 1.0)):
            for index in range(4):
                episode_id = f"{dataset}-{outcome_name}-{index}"
                src = f"videos/{episode_id}.mp4"
                (input_root / src).write_bytes(episode_id.encode())
                videos.append(
                    {
                        "src": src,
                        "poster": "",
                        "info": "generic instruction<br><br>private details",
                        "camera_id": "camera-front",
                        "camera_label": "front",
                        "task_id": f"task-{outcome_name}",
                        "task_label": outcome_name.title(),
                        "annotation_id": "annotation",
                        "annotation_label": "annotation",
                        "failure_category_ids": [],
                        "failure_category_labels": [],
                        "severity_id": "severity",
                        "severity_label": "severity",
                        "failure_description": "",
                        "additional_notes": "",
                        "episode_id": episode_id,
                        "success": success,
                        "dataset_repo": f"OopsieData-Submissions/{dataset}",
                        "bulk_import": False,
                    }
                )

    (input_root / "metadata.json").write_text(json.dumps({"videos": videos}))
    missing_faces = tmp_path / "missing-faces.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_public_visualizer_subset",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--face-episodes-csv",
            str(missing_faces),
            "--face-videos-csv",
            str(missing_faces),
            "--n-failure",
            "6",
            "--n-success",
            "3",
            "--min-failure-per-dataset",
            "1",
            "--min-success-per-dataset",
            "1",
            "--dataset-weight-power",
            "0.5",
        ],
    )

    exporter.main()

    public_metadata = json.loads((output_root / "metadata.json").read_text())
    source = public_metadata["source"]
    assert len(public_metadata["videos"]) == 9
    assert sum(video["success"] < 0.5 for video in public_metadata["videos"]) == 6
    assert sum(video["success"] >= 0.5 for video in public_metadata["videos"]) == 3
    assert source["dataset_sampling"] == {
        "strategy": "outcome-weighted",
        "weight_power": 0.5,
        "min_failure_per_dataset": 1,
        "min_success_per_dataset": 1,
        "represented_datasets": 3,
    }
    serialized = json.dumps(public_metadata)
    assert "OopsieData-Submissions" not in serialized
    assert all(dataset not in serialized for dataset in ("alpha", "beta", "gamma"))


def test_equal_total_quotas_then_fit_exact_outcomes() -> None:
    failures = (
        make_episodes("a", 10)
        + make_episodes("b", 10)
        + make_episodes("small", 2)
    )
    successes = (
        make_episodes("a", 10, success=1.0)
        + make_episodes("b", 10, success=1.0)
    )
    total_quotas = exporter.allocate_dataset_quotas(
        failures + successes,
        target=12,
        minimum=0,
        weight_power=0,
    )

    failure_quotas, success_quotas = (
        exporter.allocate_outcomes_within_total_quotas(
            failures,
            successes,
            total_quotas,
            target_success=4,
        )
    )

    assert total_quotas == {"a": 5, "b": 5, "small": 2}
    assert sum(failure_quotas.values()) == 8
    assert sum(success_quotas.values()) == 4
    assert {
        name: failure_quotas[name] + success_quotas[name]
        for name in total_quotas
    } == total_quotas


def test_outcome_allocation_respects_dataset_capacities() -> None:
    failures = make_episodes("failure_only", 10)
    successes = make_episodes("success_only", 10, success=1.0)

    failure_quotas, success_quotas = (
        exporter.allocate_outcomes_within_total_quotas(
            failures,
            successes,
            {"failure_only": 5, "success_only": 5},
            target_success=5,
        )
    )

    assert failure_quotas == {"failure_only": 5, "success_only": 0}
    assert success_quotas == {"failure_only": 0, "success_only": 5}


def test_outcome_allocation_rejects_infeasible_success_target() -> None:
    failures = make_episodes("a", 10)
    successes = make_episodes("a", 1, success=1.0)

    with pytest.raises(ValueError, match="feasible range"):
        exporter.allocate_outcomes_within_total_quotas(
            failures,
            successes,
            {"a": 5},
            target_success=2,
        )


def test_rebalance_total_quotas_lowers_forced_success_minimum() -> None:
    failures = make_episodes("failure_rich", 10)
    successes = make_episodes("success_rich", 10, success=1.0)

    quotas = exporter.rebalance_total_quotas_for_outcome_target(
        failures,
        successes,
        {"failure_rich": 4, "success_rich": 6},
        target_success=4,
    )

    assert quotas == {"failure_rich": 6, "success_rich": 4}
    failure_quotas, success_quotas = (
        exporter.allocate_outcomes_within_total_quotas(
            failures,
            successes,
            quotas,
            target_success=4,
        )
    )
    assert sum(failure_quotas.values()) == 6
    assert sum(success_quotas.values()) == 4


def test_rebalance_total_quotas_raises_success_capacity() -> None:
    failures = make_episodes("failure_rich", 10)
    successes = make_episodes("success_rich", 10, success=1.0)

    quotas = exporter.rebalance_total_quotas_for_outcome_target(
        failures,
        successes,
        {"failure_rich": 7, "success_rich": 3},
        target_success=5,
    )

    assert quotas == {"failure_rich": 5, "success_rich": 5}
