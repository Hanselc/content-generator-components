"""Generate speech audio from text using VoxCPM Hi-Fi Cloning.

Core function: generate_audio(...)
CLI: python tts.py "<text>" --name-prefix 0_ --output-folder /path/to/folder

Uses the VoxCPM2 model with Hi-Fi Cloning: a reference audio clip and its
exact transcript are provided so the model reproduces every vocal nuance.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf


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


DEFAULT_CFG_VALUE = _env_float("VOXCPM_CFG_VALUE", 2.0)
DEFAULT_INFERENCE_TIMESTEPS = _env_int("VOXCPM_INFERENCE_TIMESTEPS", 10)
DEFAULT_NORMALIZE = False
DEFAULT_MAX_LEN = _env_int("VOXCPM_MAX_LEN", 2000)
DEFAULT_RETRY_BADCASE = _env_bool("VOXCPM_RETRY_BADCASE", False)
MAX_SEGMENT_CHARS = _env_int("VOXCPM_MAX_SEGMENT_CHARS", 200)

_model = None
_model_name = None


def _get_model(model_name: str):
    """Lazy-load and cache the VoxCPM model (singleton per process)."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from voxcpm import VoxCPM

        device = os.environ.get("VOXCPM_DEVICE", "auto")
        load_denoiser = _env_bool("VOXCPM_LOAD_DENOISER", False)
        optimize = _env_bool("VOXCPM_OPTIMIZE", False)
        _model = VoxCPM.from_pretrained(
            model_name,
            device=device,
            load_denoiser=load_denoiser,
            optimize=optimize,
        )
        _model_name = model_name
    return _model


def _clear_cuda_cache():
    """Release fragmented CUDA memory between generations."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


_SENTENCE_ENDS = ".!?。！？"
_CLAUSE_ENDS = ",;:、，；："
_CONJUNCTIONS = [
    r"\by\b", r"\bo\b", r"\bpero\b", r"\bporque\b", r"\baunque\b",
    r"\bsino\b", r"\bmientras\b", r"\bcuando\b", r"\bpara que\b",
    r"\band\b", r"\bbut\b", r"\bor\b", r"\bbecause\b",
    r"\balthough\b", r"\bwhen\b", r"\bso that\b",
]


def _split_at_punctuation(text: str, delimiters: str) -> list[str]:
    """Split text after each delimiter character, keeping it attached."""
    pattern = r"(?<=[" + re.escape(delimiters) + r"])\s*"
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]


def _split_long_segment(segment: str, max_chars: int) -> list[str]:
    """Split a segment that exceeds max_chars at the best available boundary."""
    if len(segment) <= max_chars:
        return [segment]

    # 1. Try clause boundaries (commas, semicolons, colons).
    clauses = _split_at_punctuation(segment, _CLAUSE_ENDS)
    if len(clauses) > 1:
        result: list[str] = []
        current = ""
        for clause in clauses:
            candidate = (current + " " + clause).strip() if current else clause
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(clause) <= max_chars:
                    current = clause
                else:
                    result.extend(_split_long_segment(clause, max_chars))
                    current = ""
        if current:
            result.append(current)
        return result

    # 2. Try conjunctions (split before the conjunction).
    conj_pattern = r"(\s+(?:" + "|".join(_CONJUNCTIONS) + r")\b)"
    conj_parts = re.split(conj_pattern, segment, flags=re.IGNORECASE)
    if len(conj_parts) > 1:
        # re.split with a capturing group produces: [text, sep, text, sep, text, ...]
        # Reassemble so each conjunction stays attached to the text that follows it.
        pieces: list[str] = []
        for i in range(0, len(conj_parts), 2):
            piece = conj_parts[i]
            if i + 1 < len(conj_parts):
                piece += conj_parts[i + 1]
            pieces.append(piece)
        result = []
        current = ""
        for piece in pieces:
            candidate = (current + piece).strip() if current else piece.strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(piece.strip()) <= max_chars:
                    current = piece.strip()
                else:
                    result.extend(_split_long_segment(piece.strip(), max_chars))
                    current = ""
        if current:
            result.append(current)
        return result

    # 3. Last resort: split at the nearest word boundary near max_chars.
    words = segment.split()
    if len(words) <= 1:
        return [segment]
    result = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip() if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                result.append(current)
            current = word
    if current:
        result.append(current)
    return result


def _split_text(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split text into segments suitable for stable VoxCPM generation.

    Splitting priority (each keeps the delimiter attached):
      1. Sentence-ending punctuation (. ! ? 。 ！ ？)
      2. If a sentence exceeds max_chars: clause boundaries (, ; : 、 ， ；)
      3. If still too long: conjunctions (y, o, pero, and, but, ...)
      4. Last resort: word boundaries near max_chars

    VoxCPM can drift on long inputs, so we segment and generate each piece
    independently, then concatenate the waveforms into a single output.
    """
    text = text.strip()
    if not text:
        return []

    # 1. Split on sentence-ending punctuation.
    sentences = _split_at_punctuation(text, _SENTENCE_ENDS)
    if not sentences:
        sentences = [text]

    # 2. Split any sentence that exceeds max_chars.
    segments: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            segments.append(sentence)
        else:
            segments.extend(_split_long_segment(sentence, max_chars))

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
    max_len: int = DEFAULT_MAX_LEN,
    streaming: bool = False,
    retry_badcase: bool = DEFAULT_RETRY_BADCASE,
) -> dict:
    """Synthesize speech with Hi-Fi Cloning and write a .wav file.

    prompt_wav_path + prompt_text provide the prompt pair (exact transcript)
    used for alignment/continuation. reference_wav_path provides a separate
    clip that reinforces the timbre; if None, prompt_wav_path is reused.

    max_len caps the autoregressive token loop (lower = less VRAM).
    streaming keeps a rolling patch window instead of accumulating all
    generated patches (lowers peak VRAM for long outputs).
    retry_badcase re-runs inference on abnormal outputs (disabling avoids
    repeated peak-memory spikes).

    Returns a dict with audio metadata:
      sample_rate, duration_seconds, file_size_bytes, model, mode, created_at
    """
    output_path = Path(output_path)
    prompt_wav_path = str(prompt_wav_path)
    reference_wav_path = str(reference_wav_path) if reference_wav_path else prompt_wav_path

    model = _get_model(model_name)
    sample_rate = model.tts_model.sample_rate

    segments = _split_text(text)
    if not segments:
        raise ValueError("text is empty or contains no speakable content")

    all_wavs: list[np.ndarray] = []
    for seg in segments:
        if streaming:
            chunks: list[np.ndarray] = []
            for chunk in model.generate_streaming(
                text=seg,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                reference_wav_path=reference_wav_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                max_len=max_len,
                normalize=normalize,
                retry_badcase=retry_badcase,
            ):
                chunks.append(np.asarray(chunk, dtype=np.float32))
            wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        else:
            wav = model.generate(
                text=seg,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                reference_wav_path=reference_wav_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                max_len=max_len,
                normalize=normalize,
                retry_badcase=retry_badcase,
            )
            wav = np.asarray(wav, dtype=np.float32)
        all_wavs.append(wav)
        _clear_cuda_cache()

    full_wav = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), full_wav, sample_rate)
    _clear_cuda_cache()

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
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN,
                        help=f"Max generated audio tokens per segment (default {DEFAULT_MAX_LEN})")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming generation (lower peak VRAM for long text)")
    parser.add_argument("--no-retry-badcase", dest="retry_badcase",
                        action="store_false", default=DEFAULT_RETRY_BADCASE,
                        help="Disable bad-case retries (default: %s)" % DEFAULT_RETRY_BADCASE)
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    ref_path = Path(args.reference_json) if args.reference_json else script_dir / "reference.json"
    if not ref_path.is_file():
        print(json.dumps({"error": f"reference.json not found: {ref_path}"}), file=sys.stderr)
        return 1
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    prompt_audio_rel = ref.get("prompt_audio")
    prompt_text = ref.get("prompt_text")
    if not prompt_audio_rel or not prompt_text:
        print(json.dumps({"error": "reference.json must contain 'prompt_audio' and 'prompt_text'"}), file=sys.stderr)
        return 1
    prompt_wav = (script_dir / prompt_audio_rel).resolve()
    if not prompt_wav.is_file():
        print(json.dumps({"error": f"prompt audio not found: {prompt_wav}"}), file=sys.stderr)
        return 1

    reinforce_enabled = ref.get("reinforce_enabled", False)
    reference_wav_path = None
    if reinforce_enabled:
        reinforce_audio_rel = ref.get("reinforce_audio")
        if not reinforce_audio_rel:
            print(json.dumps({"error": "reference.json: 'reinforce_audio' required when reinforce_enabled is true"}), file=sys.stderr)
            return 1
        reinforce_wav = (script_dir / reinforce_audio_rel).resolve()
        if not reinforce_wav.is_file():
            print(json.dumps({"error": f"reinforce audio not found: {reinforce_wav}"}), file=sys.stderr)
            return 1
        reference_wav_path = str(reinforce_wav)

    hhmmss = datetime.now().strftime("%H%M%S")
    filename = f"{args.name_prefix}{hhmmss}.wav"
    output_path = Path(args.output_folder).resolve() / filename

    try:
        result = generate_audio(
            text=args.text,
            prompt_wav_path=str(prompt_wav),
            prompt_text=prompt_text,
            output_path=output_path,
            reference_wav_path=reference_wav_path,
            model_name=args.model,
            cfg_value=args.cfg,
            inference_timesteps=args.timesteps,
            normalize=args.normalize,
            max_len=args.max_len,
            streaming=args.streaming,
            retry_badcase=args.retry_badcase,
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

    print(json.dumps({"audio_path": filename, "metadata": result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())