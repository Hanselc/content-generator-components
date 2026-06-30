"""HTTP wrapper around tts.generate_audio.

Run (dev):
    python server.py
Run (prod):
    gunicorn -w 1 -b 0.0.0.0:8788 server:app

The only job of this service is: read input text, write an audio file.
Voices are selected by referenceId; each referenceId maps to a subfolder under
references/ containing its own reference.json + audio clips.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from tts import (
    DEFAULT_CFG_VALUE,
    DEFAULT_MAX_GENERATE_LENGTH,
    DEFAULT_TEMPERATURE,
    generate_audio,
    is_model_loaded,
    teardown,
)

# Load .env from this script's directory.
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

# Directory holding per-voice subfolders (references/<referenceId>/).
REFERENCES_DIR = Path(os.environ.get("REFERENCES_DIR", "")).expanduser() if os.environ.get("REFERENCES_DIR") else SCRIPT_DIR / "references"
REFERENCES_DIR = REFERENCES_DIR.resolve()

# Model identifier (HuggingFace hub name or local path). One model serves all
# voices; only the encoded prompt/reference latents differ per voice.
MODEL_NAME = os.environ.get("MODEL_NAME", "openbmb/VoxCPM2")

# Default referenceId used when a request omits it.
DEFAULT_REFERENCE_ID = "default"


def _list_reference_ids() -> list[str]:
    """List referenceIds that have a reference.json under REFERENCES_DIR."""
    if not REFERENCES_DIR.is_dir():
        return []
    ids = []
    for entry in sorted(REFERENCES_DIR.iterdir()):
        if entry.is_dir() and (entry / "reference.json").is_file():
            ids.append(entry.name)
    return ids


def _load_reference(reference_id: str) -> dict:
    """Load and validate references/<reference_id>/reference.json.

    Returns a dict with prompt_wav_path, prompt_text, and optionally
    reference_wav_path. Raises ValueError on missing files or invalid fields.
    """
    ref_dir = REFERENCES_DIR / reference_id
    ref_path = ref_dir / "reference.json"
    if not ref_path.is_file():
        raise ValueError(f"referenceId not found: {reference_id!r} (no reference.json at {ref_path})")

    ref = json.loads(ref_path.read_text(encoding="utf-8"))

    prompt_audio_rel = ref.get("prompt_audio")
    prompt_text = ref.get("prompt_text")
    if not isinstance(prompt_audio_rel, str) or not prompt_audio_rel:
        raise ValueError(f"reference.json for {reference_id!r}: 'prompt_audio' is required and must be a string")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError(f"reference.json for {reference_id!r}: 'prompt_text' is required and must be a string")

    prompt_audio_path = (ref_dir / prompt_audio_rel).resolve()
    if not prompt_audio_path.is_file():
        raise ValueError(f"prompt audio file not found: {prompt_audio_path}")

    result = {"prompt_wav_path": str(prompt_audio_path), "prompt_text": prompt_text}

    reinforce_enabled = ref.get("reinforce_enabled", False)
    if reinforce_enabled:
        reinforce_audio_rel = ref.get("reinforce_audio")
        if not isinstance(reinforce_audio_rel, str) or not reinforce_audio_rel:
            raise ValueError(f"reference.json for {reference_id!r}: 'reinforce_audio' is required when reinforce_enabled is true")
        reinforce_audio_path = (ref_dir / reinforce_audio_rel).resolve()
        if not reinforce_audio_path.is_file():
            raise ValueError(f"reinforce audio file not found: {reinforce_audio_path}")
        result["reference_wav_path"] = str(reinforce_audio_path)

    return result


app = Flask(__name__)


@app.get("/health")
def health():
    """Report service health only (no voice/model state)."""
    if not REFERENCES_DIR.is_dir():
        return jsonify({"status": "error", "detail": f"references dir not found: {REFERENCES_DIR}"}), 500
    ids = _list_reference_ids()
    if not ids:
        return jsonify({"status": "error", "detail": f"no reference voices found under {REFERENCES_DIR}"}), 500
    return jsonify({"status": "ok"})


@app.get("/references")
def list_references():
    """List available referenceIds."""
    return jsonify({"references": _list_reference_ids()})


@app.post("/release")
def release_model():
    """Release the loaded model and free GPU/CPU memory.

    The model is lazy-loaded again on the next /generate request.
    """
    teardown()
    return jsonify({"status": "released", "model_loaded": is_model_loaded()})


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    output_path = data.get("output_path")
    reference_id = data.get("referenceId") or DEFAULT_REFERENCE_ID

    if not text or not str(text).strip():
        return jsonify({"error": "text is required and must be non-empty"}), 400
    if not output_path or not isinstance(output_path, str):
        return jsonify({"error": "output_path is required"}), 400
    if not os.path.isabs(output_path):
        return jsonify({"error": "output_path must be an absolute path"}), 400

    try:
        ref = _load_reference(reference_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    cfg_value = float(data.get("cfg_value", DEFAULT_CFG_VALUE))
    temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
    max_generate_length = int(data.get("max_generate_length", DEFAULT_MAX_GENERATE_LENGTH))

    try:
        result = generate_audio(
            text=text,
            prompt_wav_path=ref["prompt_wav_path"],
            prompt_text=ref["prompt_text"],
            output_path=output_path,
            reference_wav_path=ref.get("reference_wav_path"),
            model_name=MODEL_NAME,
            cfg_value=cfg_value,
            temperature=temperature,
            max_generate_length=max_generate_length,
            reference_id=reference_id,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"audio generation failed: {e}"}), 500

    response = {"audio_path": output_path, "reference_id": reference_id, "metadata": result}
    return jsonify(response), 200


if __name__ == "__main__":
    import argparse

    default_port = int(os.environ.get("PORT", 8788))
    parser = argparse.ArgumentParser(description="Vox-CPM text-to-speech HTTP server.")
    parser.add_argument("--port", type=int, default=default_port, help="Port to listen on (default: %(default)s).")
    parser.add_argument("--listen", type=str, default="0.0.0.0", help="Host/IP to listen on (default: %(default)s).")
    args = parser.parse_args()
    app.run(host=args.listen, port=args.port)