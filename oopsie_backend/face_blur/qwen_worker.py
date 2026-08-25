"""JSON-lines Qwen2.5-VL image classifier worker.

The worker deliberately has no face-detector dependency. A parent process
encodes detector-positive full frames as JPEGs, sends batches over stdin, and
receives the model's raw text responses over stdout. Model and library logs go
to stderr so stdout remains machine-readable.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model id or local snapshot path.",
    )
    parser.add_argument(
        "--revision", default="cc594898137f460bfe9f0759e9844b3ce807cfb5"
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-visual-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens", type=int, default=1280)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _load_runtime(args: argparse.Namespace):
    try:
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Qwen verification requires torch, transformers, accelerate, and Pillow; "
            f"underlying import error: {exc}"
        ) from exc

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("Qwen CUDA device requested, but torch cannot access CUDA")
    if args.min_visual_tokens <= 0:
        raise ValueError("--min-visual-tokens must be positive")
    if args.max_visual_tokens < args.min_visual_tokens:
        raise ValueError("--max-visual-tokens must be at least --min-visual-tokens")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    cache_dir = str(args.cache_dir.resolve()) if args.cache_dir else None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        min_pixels=args.min_visual_tokens * 28 * 28,
        max_pixels=args.max_visual_tokens * 28 * 28,
    )
    # Qwen is a decoder-only model. Batched generation must left-pad prompts;
    # right padding makes shorter examples continue from pad tokens and can
    # yield unrelated completions even when the first item in a batch is sane.
    processor.tokenizer.padding_side = "left"
    model.generation_config.temperature = None
    metadata = {
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "device": args.device,
        "dtype": str(model.dtype),
        "attention": "sdpa",
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "max_new_tokens": args.max_new_tokens,
        "padding_side": processor.tokenizer.padding_side,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return torch, Image, model, processor, metadata


def _decode_images(Image, encoded_images: list[str]):
    images = []
    for encoded in encoded_images:
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(payload)) as image:
            images.append(image.convert("RGB"))
    return images


def _classify_batch(
    *,
    torch,
    Image,
    model,
    processor,
    prompt: str,
    encoded_images: list[str],
    max_new_tokens: int,
) -> list[str]:
    images = _decode_images(Image, encoded_images)
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image in images
    ]
    texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _write_message(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        torch, Image, model, processor, metadata = _load_runtime(args)
    except Exception as exc:  # noqa: BLE001 - JSON-lines protocol boundary
        _write_message(
            {
                "type": "ready",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return 1

    _write_message({"type": "ready", "status": "ok", "metadata": metadata})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            if request.get("command") == "shutdown":
                _write_message(
                    {"type": "shutdown", "status": "ok", "request_id": request_id}
                )
                return 0
            prompt = str(request["prompt"])
            encoded_images = list(request["images"])
            if not encoded_images:
                raise ValueError("request contains no images")
            raw_responses = _classify_batch(
                torch=torch,
                Image=Image,
                model=model,
                processor=processor,
                prompt=prompt,
                encoded_images=encoded_images,
                max_new_tokens=args.max_new_tokens,
            )
            _write_message(
                {
                    "type": "result",
                    "status": "ok",
                    "request_id": request_id,
                    "raw_responses": raw_responses,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep the worker alive per request
            _write_message(
                {
                    "type": "result",
                    "status": "error",
                    "request_id": request_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
