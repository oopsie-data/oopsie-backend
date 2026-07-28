"""Pinned, checksum-verified face detector model provisioning.

Only detector ONNX files are retained. InsightFace release archives also hold
recognition and attribute models; those members are never extracted.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    download_sha256: str
    model_sha256: str
    source: str
    license_url: str
    archive_member: str | None = None


INSIGHTFACE_LICENSE = "https://github.com/deepinsight/insightface#license"
YUNET_LICENSE = (
    "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"
)
DEFACE_LICENSE = "https://github.com/ORB-HD/deface/blob/v1.5.0/LICENSE"

MODEL_SPECS: dict[str, ModelSpec] = {
    "yunet": ModelSpec(
        name="yunet",
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/"
            "f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/"
            "face_detection_yunet_2023mar.onnx"
        ),
        download_sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        model_sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        source="OpenCV Zoo YuNet 2023mar at f12e12798e8314f7c074a6656816c048dcc95b7a",
        license_url=YUNET_LICENSE,
    ),
    "scrfd-500m": ModelSpec(
        name="scrfd-500m",
        filename="scrfd_500m.onnx",
        url=(
            "https://github.com/deepinsight/insightface/releases/download/"
            "v0.7/buffalo_sc.zip"
        ),
        download_sha256="57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72",
        model_sha256="5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
        archive_member="det_500m.onnx",
        source="InsightFace v0.7 buffalo_sc / SCRFD 500M",
        license_url=INSIGHTFACE_LICENSE,
    ),
    "scrfd-2.5g": ModelSpec(
        name="scrfd-2.5g",
        filename="scrfd_2.5g.onnx",
        url=(
            "https://github.com/deepinsight/insightface/releases/download/"
            "v0.7/buffalo_m.zip"
        ),
        download_sha256="d98264bd8f2dc75cbc2ddce2a14e636e02bb857b3051c234b737bf3b614edca9",
        model_sha256="041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9",
        archive_member="buffalo_m/det_2.5g.onnx",
        source="InsightFace v0.7 buffalo_m / SCRFD 2.5G",
        license_url=INSIGHTFACE_LICENSE,
    ),
    "scrfd-10g": ModelSpec(
        name="scrfd-10g",
        filename="scrfd_10g.onnx",
        url=(
            "https://github.com/deepinsight/insightface/releases/download/"
            "v0.7/buffalo_l.zip"
        ),
        download_sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
        model_sha256="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        archive_member="det_10g.onnx",
        source="InsightFace v0.7 buffalo_l / SCRFD 10G",
        license_url=INSIGHTFACE_LICENSE,
    ),
    "scrfd-34g": ModelSpec(
        name="scrfd-34g",
        filename="scrfd_34g.onnx",
        url=(
            "https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX/"
            "resolve/main/34g.onnx"
        ),
        download_sha256="5fb6dce6dbdb3a2866ec401297fca008fbd48ea768ed2edb33546a8d7c3ae39d",
        model_sha256="5fb6dce6dbdb3a2866ec401297fca008fbd48ea768ed2edb33546a8d7c3ae39d",
        source=(
            "InsightFace SCRFD 34G pretrained detector converted to ONNX "
            "by RuteNL/SCRFD-face-detection-ONNX"
        ),
        license_url=INSIGHTFACE_LICENSE,
    ),
    "deface-centerface": ModelSpec(
        name="deface-centerface",
        filename="deface_centerface_v1.5.0.onnx",
        url=(
            "https://raw.githubusercontent.com/ORB-HD/deface/"
            "v1.5.0/deface/centerface.onnx"
        ),
        download_sha256=(
            "09189deaaf8646c5c51a68447e3c744ea1e211798155d4728c20507b9f5aefbc"
        ),
        model_sha256=(
            "09189deaaf8646c5c51a68447e3c744ea1e211798155d4728c20507b9f5aefbc"
        ),
        source="ORB-HD/deface v1.5.0 bundled CenterFace model",
        license_url=DEFACE_LICENSE,
    ),
}


def default_model_dir() -> Path:
    configured = os.environ.get("OOPSIE_FACE_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path("face_blur_models")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "oopsie-backend-face-detector/1"},
    )
    with urllib.request.urlopen(  # noqa: S310 - pinned HTTPS URL
        request, timeout=120
    ) as response:
        length = response.headers.get("Content-Length")
        total = int(length) if length else None
        with (
            destination.open("wb") as fh,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=destination.name,
                file=sys.stderr,
            ) as progress,
        ):
            while chunk := response.read(1024 * 1024):
                fh.write(chunk)
                progress.update(len(chunk))


def provision_model(name: str, model_dir: Path | None = None) -> Path:
    """Return a verified model path, downloading and extracting if needed."""
    try:
        spec = MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown model {name!r}; choose from {', '.join(MODEL_SPECS)}"
        ) from exc

    root = (model_dir or default_model_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / spec.filename
    if target.is_file() and sha256_file(target) == spec.model_sha256:
        return target
    target.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as temp_name:
        temp_dir = Path(temp_name)
        # Keep the downloaded artifact distinct from the extracted/final model;
        # direct ONNX URLs often already use ``spec.filename``.
        downloaded = temp_dir / "download"
        print(f"Downloading {spec.source}", file=sys.stderr)
        _download(spec.url, downloaded)
        actual_download_hash = sha256_file(downloaded)
        if actual_download_hash != spec.download_sha256:
            raise RuntimeError(
                f"checksum mismatch for {spec.url}: expected "
                f"{spec.download_sha256}, got {actual_download_hash}"
            )

        staged = temp_dir / spec.filename
        if spec.archive_member:
            with zipfile.ZipFile(downloaded) as archive:
                try:
                    member = archive.open(spec.archive_member)
                except KeyError as exc:
                    raise RuntimeError(
                        f"archive does not contain {spec.archive_member!r}"
                    ) from exc
                with member, staged.open("wb") as output:
                    shutil.copyfileobj(member, output)
        else:
            shutil.copy2(downloaded, staged)

        actual_model_hash = sha256_file(staged)
        if actual_model_hash != spec.model_sha256:
            raise RuntimeError(
                f"model checksum mismatch for {name}: expected "
                f"{spec.model_sha256}, got {actual_model_hash}"
            )
        os.replace(staged, target)
    return target


def model_metadata(name: str, path: Path) -> dict:
    spec = MODEL_SPECS[name]
    return {
        **asdict(spec),
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        default=None,
        choices=list(MODEL_SPECS),
        help="Models to provision (default: all).",
    )
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    args = parser.parse_args(argv)
    for name in args.models or MODEL_SPECS:
        path = provision_model(name, args.model_dir)
        print(f"{name}: {path} ({sha256_file(path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
