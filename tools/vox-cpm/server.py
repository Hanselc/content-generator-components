"""HTTP wrapper around tts.generate_audio for n8n.

Run (dev):
    python server.py
Run (prod):
    gunicorn -w 1 -b 0.0.0.0:8788 server:app
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from tts import (
    DEFAULT_CFG_VALUE,
    DEFAULT_INFERENCE_TIMESTEPS,
    DEFAULT_NORMALIZE,
    generate_audio,
)

# Load .env from this script's directory.
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

# Base host path that folder names are resolved against.
WORKSPACE_BASE_PATH = os.environ.get("WORKSPACE_BASE_PATH", "")
WORKSPACE_BASE_PATH = os.path.expanduser(WORKSPACE_BASE_PATH) if WORKSPACE_BASE_PATH else ""

# Model identifier (HuggingFace hub name or local path).
MODEL_NAME = os.environ.get("MODEL_NAME", "openbmb/VoxCPM2")


def _load_reference():
    """Load and validate reference.json + reference audio at startup.

    reference.json lives in this repo alongside the server script and points
    at two reference clips plus the exact transcript of the prompt clip:
      - audio:           prompt clip used with its transcript for alignment
      - text:            exact transcript of the prompt clip
      - reference_audio: clip used to reinforce the timbre (no transcript)
    All three fields are required and shared across all requests.
    """
    ref_path = SCRIPT_DIR / "reference.json"
    if not ref_path.is_file():
        raise RuntimeError(f"reference.json not found: {ref_path}")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    audio_rel = ref.get("audio")
    prompt_text = ref.get("text")
    reference_audio_rel = ref.get("reference_audio")
    if not isinstance(audio_rel, str) or not audio_rel:
        raise RuntimeError("reference.json: 'audio' field is required and must be a string")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise RuntimeError("reference.json: 'text' field is required and must be a string")
    if not isinstance(reference_audio_rel, str) or not reference_audio_rel:
        raise RuntimeError("reference.json: 'reference_audio' field is required and must be a string")
    audio_path = (SCRIPT_DIR / audio_rel).resolve()
    if not audio_path.is_file():
        raise RuntimeError(
            f"reference audio file not found: {audio_path} — "
            f"please place the reference clip in tools/vox-cpm/references/"
        )
    reference_audio_path = (SCRIPT_DIR / reference_audio_rel).resolve()
    if not reference_audio_path.is_file():
        raise RuntimeError(
            f"reference reinforce audio file not found: {reference_audio_path} — "
            f"please place the reinforce clip in tools/vox-cpm/references/"
        )
    return {
        "prompt_wav_path": str(audio_path),
        "prompt_text": prompt_text,
        "reference_wav_path": str(reference_audio_path),
    }


try:
    REFERENCE = _load_reference()
except RuntimeError as exc:
    # Defer hard crash: store the error so /health and /make-audio can report it.
    REFERENCE = None
    _STARTUP_ERROR = str(exc)
else:
    _STARTUP_ERROR = None

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
    if _STARTUP_ERROR is not None:
        return jsonify({"status": "error", "detail": _STARTUP_ERROR}), 500
    return jsonify({"status": "ok"})


@app.post("/make-audio")
def make_audio():
    if _STARTUP_ERROR is not None:
        return jsonify({"error": f"server not ready: {_STARTUP_ERROR}"}), 500

    data = request.get_json(silent=True) or {}
    folder = data.get("input_folder") or data.get("folder")
    text = data.get("text")
    name_prefix = data.get("name_prefix")

    if not folder:
        return jsonify({"error": "input_folder (or folder) is required"}), 400
    if not text or not str(text).strip():
        return jsonify({"error": "text is required and must be non-empty"}), 400
    if not name_prefix:
        return jsonify({"error": "name_prefix is required"}), 400

    try:
        input_folder = resolve_input_folder(folder)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not os.path.isdir(input_folder):
        return jsonify({"error": f"input_folder does not exist: {input_folder}"}), 404

    cfg_value = float(data.get("cfg_value", DEFAULT_CFG_VALUE))
    inference_timesteps = int(data.get("inference_timesteps", DEFAULT_INFERENCE_TIMESTEPS))
    normalize = bool(data.get("normalize", DEFAULT_NORMALIZE))

    hhmmss = datetime.now().strftime("%H%M%S")
    filename = f"{name_prefix}{hhmmss}.wav"
    output_path = os.path.join(input_folder, filename)

    try:
        result = generate_audio(
            text=text,
            prompt_wav_path=REFERENCE["prompt_wav_path"],
            prompt_text=REFERENCE["prompt_text"],
            reference_wav_path=REFERENCE["reference_wav_path"],
            output_path=output_path,
            model_name=MODEL_NAME,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"audio generation failed: {e}"}), 500

    response = {"audio_path": filename, "metadata": result}
    return jsonify(response), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8788))
    app.run(host="0.0.0.0", port=port)