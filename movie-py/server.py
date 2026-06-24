"""HTTP wrapper around make_video.build_video for n8n.

Run (dev):
    python server.py
Run (prod):
    gunicorn -w 1 -b 0.0.0.0:8080 server:app
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from make_video import build_video, DEFAULT_DISPLAY, DEFAULT_TRANSITION, MovieSpecError

# Load .env from this script's directory.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Base host path that folder names are resolved against.
WORKSPACE_BASE_PATH = os.environ.get("WORKSPACE_BASE_PATH", "")
WORKSPACE_BASE_PATH = os.path.expanduser(WORKSPACE_BASE_PATH) if WORKSPACE_BASE_PATH else ""

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
    return jsonify({"status": "ok"})


@app.post("/make-video")
def make_video():
    data = request.get_json(silent=True) or {}
    folder = data.get("input_folder") or data.get("folder")
    if not folder:
        return jsonify({"error": "input_folder (or folder) is required"}), 400

    try:
        input_folder = resolve_input_folder(folder)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    display_seconds = float(data.get("display_seconds", DEFAULT_DISPLAY))
    transition_seconds = float(data.get("transition_seconds", DEFAULT_TRANSITION))

    try:
        result = build_video(input_folder, display_seconds, transition_seconds)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except MovieSpecError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"video build failed: {e}"}), 500

    return jsonify(result), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8787))
    app.run(host="0.0.0.0", port=port)