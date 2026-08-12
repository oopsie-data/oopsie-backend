from __future__ import annotations

from oopsie_backend.scripts.export_public_visualizer_subset_from_hf import (
    apply_cmu_camera_policy,
    deduplicate_candidates,
    is_excluded_dataset_path,
    resolve_remote_video_path,
    select_camera_path,
)


def test_resolve_remote_video_path_handles_relative_and_absolute_sources() -> None:
    repo_files = {
        "Armnet/episode_000001_front.mp4",
        "Berkeley_RAIL/run/episode_000002_wrist.mp4",
    }

    assert (
        resolve_remote_video_path(
            "Armnet/episode_000001.h5",
            "episode_000001_front.mp4",
            repo_files,
        )
        == "Armnet/episode_000001_front.mp4"
    )
    assert (
        resolve_remote_video_path(
            "Berkeley_RAIL/run/episode_000002.h5",
            "/old/dataset/Berkeley_RAIL/run/episode_000002_wrist.mp4",
            repo_files,
        )
        == "Berkeley_RAIL/run/episode_000002_wrist.mp4"
    )


def test_resolve_remote_video_path_rejects_missing_or_non_video_files() -> None:
    repo_files = {"Armnet/episode_000001.h5"}

    assert (
        resolve_remote_video_path(
            "Armnet/episode_000001.h5",
            "episode_000001_front.mp4",
            repo_files,
        )
        is None
    )


def test_bulk_dataset_detection_uses_first_level_folder() -> None:
    assert is_excluded_dataset_path("roboarena_upload/episode_000001.h5")
    assert is_excluded_dataset_path("AI2_MOLMOACT2/run/episode.h5")
    assert not is_excluded_dataset_path("Armnet/episode_000001.h5")


def test_deduplicate_candidates_uses_dataset_and_episode_id() -> None:
    candidates = [
        {"dataset_name": "a", "episode_id": "1", "value": "first"},
        {"dataset_name": "a", "episode_id": "1", "value": "duplicate"},
        {"dataset_name": "b", "episode_id": "1", "value": "other dataset"},
    ]

    assert deduplicate_candidates(candidates) == [
        candidates[0],
        candidates[2],
    ]


class SelectTactile:
    def choice(self, camera_paths):
        return next(item for item in camera_paths if item[0] == "tactile")


class SelectWrist:
    def choice(self, camera_paths):
        return next(item for item in camera_paths if item[0] == "wrist")


def test_cmu_tactile_selection_is_replaced_with_front() -> None:
    selected = select_camera_path(
        {
            "dataset_name": "CMU_Intent",
            "episode_id": "episode-1",
            "camera_paths": [
                ("tactile", "CMU/tactile.mp4"),
                ("wrist", "CMU/wrist.mp4"),
                ("front", "CMU/front.mp4"),
            ],
        },
        SelectTactile(),
    )

    assert selected == ("front", "CMU/front.mp4", "tactile")


def test_cmu_front_is_preferred_over_another_visual_selection() -> None:
    selected = select_camera_path(
        {
            "dataset_name": "CMU_Intent",
            "episode_id": "episode-visual",
            "camera_paths": [
                ("wrist", "CMU/wrist.mp4"),
                ("front", "CMU/front.mp4"),
            ],
        },
        SelectWrist(),
    )

    assert selected == ("front", "CMU/front.mp4", "wrist")


def test_cmu_tactile_selection_uses_zed_when_front_is_unavailable() -> None:
    selected = select_camera_path(
        {
            "dataset_name": "CMU_Intent",
            "episode_id": "episode-2",
            "camera_paths": [
                ("tactile", "CMU/tactile.mp4"),
                ("zed", "CMU/zed.mp4"),
            ],
        },
        SelectTactile(),
    )

    assert selected == ("zed", "CMU/zed.mp4", "tactile")


def test_tactile_selection_for_other_datasets_is_unchanged() -> None:
    selected = select_camera_path(
        {
            "dataset_name": "another_dataset",
            "episode_id": "episode-3",
            "camera_paths": [
                ("tactile", "other/tactile.mp4"),
                ("front", "other/front.mp4"),
            ],
        },
        SelectTactile(),
    )

    assert selected == ("tactile", "other/tactile.mp4", None)


def test_cmu_camera_policy_removes_tactile_and_tactile_only_episodes() -> None:
    candidates = [
        {
            "dataset_name": "CMU_Intent",
            "episode_id": "front-and-tactile",
            "camera_paths": [
                ("front", "CMU/front.mp4"),
                ("tactile", "CMU/tactile.mp4"),
            ],
        },
        {
            "dataset_name": "CMU_Intent",
            "episode_id": "tactile-only",
            "camera_paths": [("tactile", "CMU/only-tactile.mp4")],
        },
        {
            "dataset_name": "other",
            "episode_id": "other-tactile",
            "camera_paths": [("tactile", "other/tactile.mp4")],
        },
    ]

    kept, removed = apply_cmu_camera_policy(candidates)

    assert removed == 1
    assert [candidate["episode_id"] for candidate in kept] == [
        "front-and-tactile",
        "other-tactile",
    ]
    assert kept[0]["camera_paths"] == [("front", "CMU/front.mp4")]
