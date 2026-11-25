import json
from pathlib import Path

import h5py
import numpy as np

from oopsie_backend.training import prepare_episode
from oopsie_backend.training.cameras import assign_roles
from oopsie_backend.training.recipe import load_recipe, unify_hash


def test_images_only_uses_index_timing_and_pads_unreadable_camera(tmp_path, monkeypatch):
    recipe = load_recipe(Path(__file__).resolve().parents[1] / "configs/training/sample_images_rlds.yaml")
    source = tmp_path / "episode.h5"
    out = tmp_path / "prepared"
    out.mkdir()
    cameras = {"main": "exterior_front", "hand": "wrist"}
    with h5py.File(source, "w") as f:
        paths = f.create_group("observations/video_paths")
        for name in cameras:
            paths[name] = f"{name}.mp4"
            (tmp_path / f"{name}.mp4").touch()

    def transcode(path, source_steps, steps, dest, size, fps, crf):
        assert source_steps == 8
        np.testing.assert_array_equal(steps, [0, 2, 4, 6])
        assert (fps, size) == (10, (180, 320))
        if path.stem == "hand":
            raise prepare_episode.Skip("no frames")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    monkeypatch.setattr(prepare_episode, "transcode", transcode)
    unified = [{"camera": name, "name": role} for name, role in cameras.items()]
    job = ("episode", "episode.h5", source, out, recipe, unify_hash(recipe), None, unified, (8, 20))

    result = prepare_episode.unify_episode(job)

    assert result["status"] == "written"
    assert result["num_steps"] == 4
    assert json.loads(result["camera_sources"]) == {
        "wrist_1": None, "wrist_2": None, "exterior_1": "main",
    }
    assert not json.loads(result["action_widths"])
    assert not json.loads(result["state_widths"])
    assert prepare_episode.unify_episode(job)["status"] == "reused"


def test_sample_images_prefers_fixed_exterior_to_head_fallback():
    recipe = load_recipe(Path(__file__).resolve().parents[1] / "configs/training/sample_images_rlds.yaml")
    cameras = {"head": "cam_chest", "wrist_right": "right_hand"}
    assert assign_roles(cameras, recipe["cameras"]) == {
        "wrist_1": None, "wrist_2": "right_hand", "exterior_1": "cam_chest",
    }
    assert assign_roles({**cameras, "exterior_top": "ceiling"}, recipe["cameras"])["exterior_1"] == "ceiling"
