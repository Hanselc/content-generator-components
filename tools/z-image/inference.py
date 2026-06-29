"""Generate images from text prompts using Z-Image-Turbo via diffusers.

Core function: generate_image(...)
CLI: python inference.py "<prompt>" --name-prefix 0_ --output-folder /path/to/folder

The Z-Image-Turbo model is a 6B distilled single-stream DiT that produces
high-quality images in 8 NFEs. Weights are downloaded lazily on the first
call (HuggingFace cache) and the pipeline is kept in memory for reuse
across sequential requests. release_resources() frees the pipeline and
clears CUDA cache so the server can reclaim GPU/RAM without restarting.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env from this script's directory so the CLI (not just server.py)
# honors WORKSPACE_BASE_PATH / ZIMAGE_* defaults.
load_dotenv(Path(__file__).resolve().parent / ".env")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


DEFAULT_WIDTH = _env_int("ZIMAGE_WIDTH", 1024)
DEFAULT_HEIGHT = _env_int("ZIMAGE_HEIGHT", 1024)
DEFAULT_STEPS = _env_int("ZIMAGE_STEPS", 9)
DEFAULT_CFG = _env_float("ZIMAGE_CFG", 0.0)
CPU_OFFLOAD = _env_bool("ZIMAGE_CPU_OFFLOAD", False)

_pipeline = None
_pipeline_name: str | None = None
_pipeline_device: str | None = None


def _resolve_device() -> str:
    """Resolve the runtime device from ZIMAGE_DEVICE (auto|cpu|mps|cuda|cuda:N)."""
    device = os.environ.get("ZIMAGE_DEVICE", "auto")
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_pipeline(model_name: str, device: str):
    """Lazy-load and cache the ZImagePipeline (singleton per process).

    Weights download on the first call (HuggingFace cache). Subsequent
    calls reuse the cached pipeline as long as model_name/device match.
    """
    global _pipeline, _pipeline_name, _pipeline_device
    if _pipeline is None or _pipeline_name != model_name or _pipeline_device != device:
        import torch
        from diffusers import ZImagePipeline

        dtype = torch.bfloat16 if "cuda" in device or "mps" in device else torch.float32
        _pipeline = ZImagePipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        if CPU_OFFLOAD:
            # Offload at the submodule level: each transformer/VAE/CLIP sub-layer
            # moves to GPU only for its forward pass, then back to CPU. This
            # keeps peak VRAM low enough to run the 6B DiT on 8 GB cards, at
            # the cost of significant host<->device transfer overhead per step.
            _pipeline.enable_sequential_cpu_offload()
        else:
            _pipeline.to(device)
        _pipeline_name = model_name
        _pipeline_device = device
    return _pipeline


def _clear_cuda_cache():
    """Release fragmented CUDA memory between generations."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _cuda_mem_info() -> dict | None:
    """Return CUDA memory stats if available, else None."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {
                "free_bytes": free,
                "total_bytes": total,
                "allocated_bytes": total - free,
            }
    except Exception:
        pass
    return None


def is_model_loaded() -> bool:
    """Whether the pipeline is currently loaded in memory."""
    return _pipeline is not None


def release_resources() -> dict:
    """Drop the cached pipeline and reclaim GPU/RAM.

    Resets the singleton so the next generate_image() call reloads the
    model. Runs gc.collect() and torch.cuda.empty_cache() to release
    fragmented GPU memory. Safe to call when nothing is loaded.
    """
    global _pipeline, _pipeline_name, _pipeline_device
    was_loaded = _pipeline is not None
    before = _cuda_mem_info()

    _pipeline = None
    _pipeline_name = None
    _pipeline_device = None

    gc.collect()
    _clear_cuda_cache()

    after = _cuda_mem_info()

    result: dict = {
        "was_loaded": was_loaded,
        "freed_cuda_bytes": None,
        "cuda_after": after,
    }
    if before is not None and after is not None:
        result["freed_cuda_bytes"] = before["allocated_bytes"] - after["allocated_bytes"]
    return result


def generate_image(
    prompt: str,
    output_path: str | os.PathLike,
    model_name: str = "Tongyi-MAI/Z-Image-Turbo",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    seed: int | None = None,
    negative_prompt: str | None = None,
) -> dict:
    """Synthesize an image from a text prompt and write a .png file.

    For Z-Image-Turbo the recommended settings are: steps=9 (yields 8 DiT
    forwards), cfg=0.0 (Turbo is guidance-free). The non-Turbo Z-Image
    model benefits from cfg=3.0-5.0, steps=28-50, and a negative_prompt.

    Returns a dict with image metadata:
      width, height, steps, cfg, seed, model, created_at, file_size_bytes
    """
    import torch

    output_path = Path(output_path)
    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError("prompt is required and must be non-empty")

    device = _resolve_device()
    pipe = _get_pipeline(model_name, device)

    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    generator = torch.Generator(device).manual_seed(int(seed))

    kwargs: dict = dict(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        generator=generator,
    )
    if negative_prompt:
        kwargs["negative_prompt"] = str(negative_prompt)

    try:
        image = pipe(**kwargs).images[0]
    finally:
        _clear_cuda_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path), format="PNG")
    _clear_cuda_cache()

    file_size = output_path.stat().st_size

    return {
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "cfg": float(cfg),
        "seed": int(seed),
        "model": model_name,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file_size_bytes": file_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an image via Z-Image-Turbo.")
    parser.add_argument("prompt", help="Text prompt to synthesize")
    parser.add_argument("--name-prefix", default="0_",
                        help="Filename prefix (e.g. '0_'). Default: '0_'")
    parser.add_argument("--output-folder", required=True,
                        help="Folder to write the generated .png into")
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo",
                        help="Model name or local path")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help=f"Output width (default {DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                        help=f"Output height (default {DEFAULT_HEIGHT})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Inference steps (default {DEFAULT_STEPS})")
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG,
                        help=f"CFG guidance scale (default {DEFAULT_CFG})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: random)")
    parser.add_argument("--negative-prompt", default=None,
                        help="Negative prompt (ignored by Turbo; useful for Z-Image)")
    args = parser.parse_args(argv)

    hhmmss = datetime.now().strftime("%H%M%S")
    filename = f"{args.name_prefix}{hhmmss}.png"
    output_path = Path(args.output_folder).resolve() / filename

    try:
        result = generate_image(
            prompt=args.prompt,
            output_path=output_path,
            model_name=args.model,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg=args.cfg,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

    print(json.dumps({"image_path": filename, "metadata": result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())