import hashlib
import json
from types import SimpleNamespace

import httpx
import polars as pl

from oopsie_backend.ingest import download


def test_download_uses_pinned_hashes_and_reuses_output(tmp_path, monkeypatch):
    repo = "lab/episodes"
    assets = {"episode.h5": b"episode", "videos/camera.mp4": b"video"}
    index = tmp_path / "sample.parquet"
    pl.DataFrame({
        "dataset_repo": [repo, repo],
        "repo_path": ["episode.h5", "episode.h5"],
        "h5_sha256": [hashlib.sha256(assets["episode.h5"]).hexdigest()] * 2,
        "videos": [[{
            "path": "videos/camera.mp4",
            "sha256": hashlib.sha256(assets["videos/camera.mp4"]).hexdigest(),
            "size": 999,
        }], None],
    }).write_parquet(index)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"datasets": [{"repo_id": repo, "commit": "pinned"}]}))
    requested = []

    def handle_request(request):
        path = request.url.path.removeprefix(f"/datasets/{repo}/resolve/pinned/")
        requested.append(path)
        return httpx.Response(200, content=assets[path])

    out = tmp_path / "out"
    args = ["--index", str(index), "--data-dir", str(out), "--revisions", str(snapshot), "--workers", "1"]
    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        monkeypatch.setattr(download.httpx, "stream", lambda *a, **kw: client.stream(*a, **kw))
        assert download.main(args) == 0
        assert download.main(args) == 0
    assert requested == list(assets)
    for path, content in assets.items():
        assert (out / repo / path).read_bytes() == content


def test_minimal_index_rejects_wrong_bytes_without_overwriting(tmp_path, monkeypatch):
    repo = "lab/episodes"
    index = tmp_path / "sample.parquet"
    pl.DataFrame({"dataset_repo": [repo], "repo_path": ["episode.h5"]}).write_parquet(index)
    out = tmp_path / "out"
    dest = out / repo / "episode.h5"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"existing file")
    metadata = SimpleNamespace(etag=hashlib.sha256(b"correct file").hexdigest(), commit_hash="resolved")
    monkeypatch.setattr(download, "get_hf_file_metadata", lambda *a, **kw: metadata)

    def handle_request(request):
        assert request.url.path == f"/datasets/{repo}/resolve/resolved/episode.h5"
        return httpx.Response(200, content=b"incorrect file")

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        monkeypatch.setattr(download.httpx, "stream", lambda *a, **kw: client.stream(*a, **kw))
        assert download.main(["--index", str(index), "--data-dir", str(out)]) == 1
    assert dest.read_bytes() == b"existing file"
    assert list(dest.parent.iterdir()) == [dest]
