"""Generate speech audio from text using VoxCPM Hi-Fi Cloning.

Core function: generate_audio(...)
CLI: python tts.py "<text>" --name-prefix 0_ --output-folder /path/to/folder

Uses the VoxCPM2 model with Hi-Fi Cloning: a reference audio clip and its
exact transcript are provided so the model reproduces every vocal nuance.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_CFG_VALUE = 2.0
DEFAULT_INFERENCE_TIMESTEPS = 10
DEFAULT_NORMALIZE = False

_model = None
_model_name = None


def _get_model(model_name: str):
    """Lazy-load and cache the VoxCPM model (singleton per process)."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from voxcpm import VoxCPM

        device = os.environ.get("VOXCPM_DEVICE", "auto")
        load_denoiser = os.environ.get("VOXCPM_LOAD_DENOISER", "false").lower() == "true"
        _model = VoxCPM.from_pretrained(
            model_name, device=device, load_denoiser=load_denoiser
        )
        _model_name = model_name
    return _model


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-length segments for stable long-form synthesis.

    VoxCPM can drift on long inputs, so we segment on sentence-ending /
    clause-ending punctuation and generate each piece independently, then
    concatenate the waveforms. Short inputs produce a single segment.
    """
    text = text.strip()
    if not text:
        return []
    # Split on ., !, ?, 。, ！, ？ keeping the delimiter attached.
    parts = re.split(r"(?<=[.!?。！？])\s*", text)
    segments = [p.strip() for p in parts if p.strip()]
    # If splitting produced nothing useful, fall back to the whole text.
    if not segments:
        segments = [text]
    return segments


def generate_audio(
    text: str,
    prompt_wav_path: str,
    prompt_text: str,
    output_path: str | os.PathLike,
    reference_wav_path: str | None = None,
    model_name: str = "openbmb/VoxCPM2",
    cfg_value: float = DEFAULT_CFG_VALUE,
    inference_timesteps: int = DEFAULT_INFERENCE_TIMESTEPS,
    normalize: bool = DEFAULT_NORMALIZE,
) -> dict:
    """Synthesize speech with Hi-Fi Cloning and write a .wav file.

    prompt_wav_path + prompt_text provide the prompt pair (exact transcript)
    used for alignment/continuation. reference_wav_path provides a separate
    clip that reinforces the timbre; if None, prompt_wav_path is reused.

    Returns a dict with audio metadata:
      sample_rate, duration_seconds, file_size_bytes, model, mode, created_at
    """
    output_path = Path(output_path)
    prompt_wav_path = str(prompt_wav_path)
    reference_wav_path = str(reference_wav_path) if reference_wav_path else prompt_wav_path

    model = _get_model(model_name)
    sample_rate = model.tts_model.sample_rate

    segments = _split_sentences(text)
    if not segments:
        raise ValueError("text is empty or contains no speakable content")

    all_wavs: list[np.ndarray] = []
    for seg in segments:
        wav = model.generate(
            text=seg,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            reference_wav_path=reference_wav_path,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            retry_badcase=True,
        )
        all_wavs.append(np.asarray(wav, dtype=np.float32))

    full_wav = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), full_wav, sample_rate)

    file_size = output_path.stat().st_size
    duration = len(full_wav) / float(sample_rate)

    return {
        "sample_rate": sample_rate,
        "duration_seconds": round(duration, 3),
        "file_size_bytes": file_size,
        "model": model_name,
        "mode": "hifi_clone",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate speech via VoxCPM Hi-Fi Cloning.")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--name-prefix", default="0_",
                        help="Filename prefix (e.g. '0_'). Default: '0_'")
    parser.add_argument("--output-folder", required=True,
                        help="Folder to write the generated .wav into")
    parser.add_argument("--reference-json", default=None,
                        help="Path to reference.json (default: <script dir>/reference.json)")
    parser.add_argument("--model", default="openbmb/VoxCPM2",
                        help="Model name or local path")
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG_VALUE,
                        help=f"CFG value (default {DEFAULT_CFG_VALUE})")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_INFERENCE_TIMESTEPS,
                        help=f"Inference timesteps (default {DEFAULT_INFERENCE_TIMESTEPS})")
    parser.add_argument("--normalize", action="store_true",
                        help="Enable text normalization")
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    ref_path = Path(args.reference_json) if args.reference_json else script_dir / "reference.json"
    if not ref_path.is_file():
        print(json.dumps({"error": f"reference.json not found: {ref_path}"}), file=sys.stderr)
        return 1
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    audio_rel = ref.get("audio")
    prompt_text = ref.get("text")
    reference_audio_rel = ref.get("reference_audio")
    if not audio_rel or not prompt_text:
        print(json.dumps({"error": "reference.json must contain 'audio' and 'text'"}), file=sys.stderr)
        return 1
    if not reference_audio_rel:
        print(json.dumps({"error": "reference.json must contain 'reference_audio'"}), file=sys.stderr)
        return 1
    prompt_wav = (script_dir / audio_rel).resolve()
    if not prompt_wav.is_file():
        print(json.dumps({"error": f"reference audio not found: {prompt_wav}"}), file=sys.stderr)
        return 1
    reference_wav = (script_dir / reference_audio_rel).resolve()
    if not reference_wav.is_file():
        print(json.dumps({"error": f"reference reinforce audio not found: {reference_wav}"}), file=sys.stderr)
        return 1

    hhmmss = datetime.now().strftime("%H%M%S")
    filename = f"{args.name_prefix}{hhmmss}.wav"
    output_path = Path(args.output_folder).resolve() / filename

    try:
        result = generate_audio(
            text=args.text,
            prompt_wav_path=str(prompt_wav),
            prompt_text=prompt_text,
            output_path=output_path,
            reference_wav_path=str(reference_wav),
            model_name=args.model,
            cfg_value=args.cfg,
            inference_timesteps=args.timesteps,
            normalize=args.normalize,
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

    print(json.dumps({"audio_path": filename, "metadata": result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())