"""HTTP server for movie-py.

Run (dev):
    python server.py
Run (prod):
    gunicorn -w 1 -b 0.0.0.0:8080 server:app

The server is project-agnostic: each request names a scriptId that matches a
preexisting script under ./scripts/<scriptId>/script.py. The script's build(ctx)
performs the movie construction using the shared primitives in make_video.py
and returns a section-aware result dict with time marks.

Endpoints:
    GET  /health
    GET  /scripts                  -> {"scripts": ["SocialMediaGenerator", ...]}
    GET  /scripts/<scriptId>       -> {script_id, meta, param_schema}
    POST /release                  -> no-op (API consistency with other tools)
    POST /generate                 -> build a movie (see request schema below)

POST /generate request:
    {
      "scriptId":        "SocialMediaGenerator",  # required, must preexist
      "input_folder":    "/abs/path/to/images",   # required, absolute
      "output_folder":   "/abs/path/for/output",  # required, absolute
      "spec_path":       "/abs/path/to/spec.json",# required, absolute
      "params": { ... }                           # dynamic, per-script
    }

All path fields must be absolute. `params` is validated against the script's
PARAM_SCHEMA (when declared) before build() is invoked.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from flask import Flask, jsonify, request

# Load .env from this script's directory.
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

# Directory holding per-script subfolders (scripts/<scriptId>/script.py).
SCRIPTS_DIR = SCRIPT_DIR / "scripts"

# Make the movie-py package dir importable so scripts can `import make_video`
# regardless of the current working directory.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

app = Flask(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _list_script_ids() -> list[str]:
    """List scriptIds that have a script.py under SCRIPTS_DIR."""
    if not SCRIPTS_DIR.is_dir():
        return []
    ids = []
    for entry in sorted(SCRIPTS_DIR.iterdir()):
        if entry.is_dir() and (entry / "script.py").is_file():
            ids.append(entry.name)
    return ids


def _load_script_module(script_id: str):
    """Import scripts/<scriptId>/script.py as a module and return it.

    The module is imported under the name `scripts.<scriptId>.script` so that
    re-imports during development pick up the latest code (Flask's auto-reload
    restarts the process anyway, but we use importlib.util for a clean load).
    Raises FileNotFoundError if the script file does not exist.
    """
    script_file = SCRIPTS_DIR / script_id / "script.py"
    if not script_file.is_file():
        raise FileNotFoundError(f"script not found: {script_file}")

    module_name = f"scripts.{script_id}.script"
    spec = importlib.util.spec_from_file_location(module_name, script_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load script module from {script_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Minimal JSON Schema validator (avoids a jsonschema dependency).
# Supports the subset used by our PARAM_SCHEMA declarations:
#   type, properties, required, additionalProperties, minimum, maximum,
#   enum, items (with the same subset), default (ignored on validation).
# --------------------------------------------------------------------------- #

def _validate_schema(value, schema: dict, path: str = "params") -> list[str]:
    """Return a list of validation error messages (empty if valid)."""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors  # no schema to validate against

    expected_type = schema.get("type")
    if expected_type:
        type_ok = {
            "object": dict, "array": list, "string": str,
            "number": (int, float), "integer": int,
            "boolean": bool, "null": type(None),
        }.get(expected_type)
        if type_ok is not None and not isinstance(value, type_ok):
            errors.append(f"{path}: expected type {expected_type}, got {type(value).__name__}")
            return errors

    if expected_type == "object":
        if not isinstance(value, dict):
            return errors
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}.{req}: is required")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}.{key}: additional property not allowed")
        for key, sub_schema in props.items():
            if key in value:
                errors.extend(_validate_schema(value[key], sub_schema, f"{path}.{key}"))

    elif expected_type == "array":
        if not isinstance(value, list):
            return errors
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                errors.extend(_validate_schema(item, item_schema, f"{path}[{i}]"))

    if "minimum" in schema and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}")
    if "maximum" in schema and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value > schema["maximum"]:
            errors.append(f"{path}: must be <= {schema['maximum']}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']}")

    return errors


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": False})


@app.post("/release")
def release():
    """No-op: movie-py is stateless (no GPU model to release).

    Provided for API consistency across tools so callers can call
    POST /release on any tool without checking which one it is.
    """
    return jsonify({"status": "released", "model_loaded": False})


@app.get("/scripts")
def list_scripts():
    return jsonify({"scripts": _list_script_ids()})


@app.get("/scripts/<path:script_id>")
def get_script(script_id: str):
    """Return metadata + declared PARAM_SCHEMA for a single script."""
    try:
        module = _load_script_module(script_id)
    except FileNotFoundError:
        return jsonify({"error": f"scriptId not found: {script_id!r}"}), 404
    except Exception as e:
        return jsonify({"error": f"failed to load script {script_id!r}: {e}"}), 500

    meta = getattr(module, "META", {}) or {}
    param_schema = getattr(module, "PARAM_SCHEMA", None)
    return jsonify({
        "script_id": script_id,
        "meta": meta,
        "param_schema": param_schema,
    })


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}

    # --- Common required fields -------------------------------------------
    script_id = data.get("scriptId")
    if not script_id or not isinstance(script_id, str):
        return jsonify({"error": "scriptId is required and must be a non-empty string"}), 400

    input_folder = data.get("input_folder")
    output_folder = data.get("output_folder")
    spec_path = data.get("spec_path")
    params = data.get("params", {})

    if not input_folder or not isinstance(input_folder, str):
        return jsonify({"error": "input_folder is required"}), 400
    if not output_folder or not isinstance(output_folder, str):
        return jsonify({"error": "output_folder is required"}), 400
    if not spec_path or not isinstance(spec_path, str):
        return jsonify({"error": "spec_path is required"}), 400
    if not os.path.isabs(input_folder):
        return jsonify({"error": "input_folder must be an absolute path"}), 400
    if not os.path.isabs(output_folder):
        return jsonify({"error": "output_folder must be an absolute path"}), 400
    if not os.path.isabs(spec_path):
        return jsonify({"error": "spec_path must be an absolute path"}), 400

    input_folder_p = Path(input_folder)
    spec_path_p = Path(spec_path)
    if not input_folder_p.is_dir():
        return jsonify({"error": f"input_folder does not exist or is not a directory: {input_folder}"}), 400
    if not spec_path_p.is_file():
        return jsonify({"error": f"spec_path does not exist or is not a file: {spec_path}"}), 400

    # --- Load script ------------------------------------------------------
    try:
        module = _load_script_module(script_id)
    except FileNotFoundError:
        return jsonify({"error": f"scriptId not found: {script_id!r}"}), 400
    except Exception as e:
        return jsonify({"error": f"failed to load script {script_id!r}: {e}"}), 500

    build_fn = getattr(module, "build", None)
    if not callable(build_fn):
        return jsonify({"error": f"script {script_id!r} does not expose a build(ctx) function"}), 500

    # --- Validate params against PARAM_SCHEMA (when declared) -------------
    param_schema = getattr(module, "PARAM_SCHEMA", None)
    if param_schema:
        if not isinstance(params, dict):
            return jsonify({"error": "params must be an object"}), 400
        errors = _validate_schema(params, param_schema)
        if errors:
            return jsonify({"error": "params validation failed", "details": errors}), 400
    elif params is None:
        params = {}

    # --- Build context + dispatch ----------------------------------------
    import make_video as primitives

    started_at = _now_iso()
    ctx = SimpleNamespace(
        params=params,
        input_folder=input_folder_p.resolve(),
        output_folder=Path(output_folder).resolve(),
        spec_path=spec_path_p.resolve(),
        common={
            "script_id": script_id,
            "started_at": started_at,
        },
        primitives=primitives,
    )

    try:
        result = build_fn(ctx)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except make_video.MovieSpecError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"video build failed: {e}"}), 500

    return jsonify(result), 200


if __name__ == "__main__":
    import argparse

    default_port = int(os.environ.get("PORT", 8787))
    parser = argparse.ArgumentParser(description="Movie-py video builder HTTP server.")
    parser.add_argument("--port", type=int, default=default_port,
                        help="Port to listen on (default: %(default)s).")
    parser.add_argument("--listen", type=str, default="0.0.0.0",
                        help="Host/IP to listen on (default: %(default)s).")
    args = parser.parse_args()
    app.run(host=args.listen, port=args.port)