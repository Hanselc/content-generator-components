"""HTTP wrapper around inference.generate_image for n8n.

Run (dev):
    python server.py
Run (prod):
    gunicorn -w 1 -b 0.0.0.0:8789 server:app
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from inference import (
    DEFAULT_CFG,
    DEFAULT_HEIGHT,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    generate_image,
    is_model_loaded,
    release_resources,
)

# Load .env from this script's directory.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Base host path that folder names are resolved against.
WORKSPACE_BASE_PATH = os.environ.get("WORKSPACE_BASE_PATH", "")
WORKSPACE_BASE_PATH = os.path.expanduser(WORKSPACE_BASE_PATH) if WORKSPACE_BASE_PATH else ""

# Model identifier (HuggingFace hub name or local path).
MODEL_NAME = os.environ.get("MODEL_NAME", "Tongyi-MAI/Z-Image-Turbo")

app = Flask(__name__)


def resolve_input_folder(folder_name: str) -> str:
    """Resolve a folder name (relative) against WORKSPACE_BASE_PATH.

    If folder_name is already an absolute path, it is used as-is. Otherwise it
    is joined with WORKSPACE_BASE_PATH (which must be configured).
    """
    if os.path.isabs(folder_name):
        return folder_name
    if not WORKSPACE_BASE_PATH:
        raise ValueError("relative folder name given but WORKSPACE_BASE_PATH is not set in .env")
    return os.path.join(WORKSPACE_BASE_PATH, folder_name)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": is_model_loaded()})


@app.post("/make-image")
def make_image():
    data = request.get_json(silent=True) or {}
    folder = data.get("input_folder") or data.get("folder")
    prompt = data.get("prompt")
    name_prefix = data.get("name_prefix")

    if not folder:
        return jsonify({"error": "input_folder (or folder) is required"}), 400
    if not prompt or not str(prompt).strip():
        return jsonify({"error": "prompt is required and must be non-empty"}), 400
    if not name_prefix:
        return jsonify({"error": "name_prefix is required"}), 400

    try:
        input_folder = resolve_input_folder(folder)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not os.path.isdir(input_folder):
        return jsonify({"error": f"input_folder does not exist: {input_folder}"}), 404

    width = int(data.get("width", DEFAULT_WIDTH))
    height = int(data.get("height", DEFAULT_HEIGHT))
    steps = int(data.get("steps", DEFAULT_STEPS))
    cfg = float(data.get("cfg", DEFAULT_CFG))
    seed = data.get("seed")
    seed = int(seed) if seed is not None else None
    negative_prompt = data.get("negative_prompt")

    from datetime import datetime

    hhmmss = datetime.now().strftime("%H%M%S")
    filename = f"{name_prefix}{hhmmss}.png"
    output_path = os.path.join(input_folder, filename)

    try:
        result = generate_image(
            prompt=prompt,
            output_path=output_path,
            model_name=MODEL_NAME,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            negative_prompt=negative_prompt,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"image generation failed: {e}"}), 500

    response = {"image_path": filename, "metadata": result}
    return jsonify(response), 200


@app.post("/release")
def release():
    """Free the cached pipeline and reclaim GPU/RAM.

    Drops the singleton ZImagePipeline, runs gc.collect() and
    torch.cuda.empty_cache(). The next /make-image call reloads the
    model (downloading weights again if the HF cache was purged).
    """
    try:
        result = release_resources()
    except Exception as e:
        return jsonify({"error": f"release failed: {e}"}), 500
    result["model_loaded"] = is_model_loaded()
    return jsonify(result), 200


if __name__ == "__main__":
    import argparse

    default_port = int(os.environ.get("PORT", 8789))
    parser = argparse.ArgumentParser(description="Z-Image-Turbo text-to-image HTTP server.")
    parser.add_argument("--port", type=int, default=default_port, help="Port to listen on (default: %(default)s).")
    parser.add_argument("--listen", type=str, default="0.0.0.0", help="Host/IP to listen on (default: %(default)s).")
    args = parser.parse_args()
    app.run(host=args.listen, port=args.port)