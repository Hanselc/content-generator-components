"""Generate speech audio from text using Nano-vLLM-VoxCPM Hi-Fi Cloning.

Core function: generate_audio(...)
CLI: python tts.py "<text>" --name-prefix 0_ --output-folder /path/to/folder

Uses the Nano-vLLM-VoxCPM engine with VoxCPM2 in Hi-Fi Cloning mode: a prompt
audio clip and its exact transcript are encoded to latents and passed to
server.generate(), optionally alongside a second reference clip's latents for
extra timbre fidelity. Output is 48 kHz.

This mirrors the inference flow of the reference project's VoxCPMProvider
(src/tts_providers/voxcpm/provider.py):
    snapshot_download -> VoxCPM.from_pretrained(model, devices,
        max_num_batched_tokens, max_num_seqs, gpu_memory_utilization) ->
    add_prompt(prompt_bytes, "wav", prompt_text)  [fallback: encode_latents] ->
    encode_latents(reference_bytes, "wav") ->
    iterate server.generate(target_text, prompt_id|prompt_latents, prompt_text,
                            cfg_value, temperature, max_generate_length,
                            ref_audio_latents) ->
    np.concatenate(chunks) -> sf.write(wav, 48000) -> server.stop()

NOTE: from_pretrained is called with ONLY the 4 kwargs the reference provider
passes. inference_timesteps (10), max_model_len (4096), and enforce_eager
(False) use the engine defaults — these defaults matter for the warmup peak /
KV-cache allocation balance on 8 GB GPUs. Do not override them unless you
understand the memory implications (see envs/voxcpm.yml header).
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Config (env-overridable, defaults match config.voxcpm.yaml + provider)
# ---------------------------------------------------------------------------

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


# Engine init knobs (the 4 passed to from_pretrained, matching the reference).
DEFAULT_GPU_MEMORY_UTILIZATION = _env_float("VOXCPM_GPU_MEMORY_UTILIZATION", 0.90)
DEFAULT_MAX_NUM_SEQS = _env_int("VOXCPM_MAX_NUM_SEQS", 1)
DEFAULT_MAX_NUM_BATCHED_TOKENS = _env_int("VOXCPM_MAX_NUM_BATCHED_TOKENS", 8192)
DEFAULT_DEVICE = os.environ.get("VOXCPM_DEVICE", "cuda:0")

# Per-request generation knobs.
DEFAULT_CFG_VALUE = _env_float("VOXCPM_CFG_VALUE", 2.0)
DEFAULT_TEMPERATURE = _env_float("VOXCPM_TEMPERATURE", 1.0)
DEFAULT_MAX_GENERATE_LENGTH = _env_int("VOXCPM_MAX_GENERATE_LENGTH", 2000)
MAX_SEGMENT_CHARS = _env_int("VOXCPM_MAX_SEGMENT_CHARS", 800)

# VoxCPM2 output sample rate (48 kHz, fixed by the model).
SAMPLE_RATE = 48000


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model = None
_model_name: str | None = None
_prompt_id: str | None = None
_prompt_latents_fallback: bytes | None = None
_ref_audio_latents: bytes | None = None


def _resolve_local_model(model: str) -> str:
    """Resolve a HuggingFace repo id to a local directory.

    Nano-vLLM requires a local directory with *.safetensors files. If a repo
    id is passed, download it via huggingface_hub.snapshot_download first.
    """
    p = Path(model).expanduser()
    if p.is_dir():
        return str(p)
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model)


def _read_wav_bytes(wav_path: str) -> bytes:
    with open(wav_path, "rb") as f:
        return f.read()


def _wav_duration_seconds(wav_path: str) -> float:
    """Return the duration of a wav file in seconds (0.0 on error)."""
    try:
        import soundfile as sf

        info = sf.info(wav_path)
        return float(info.duration)
    except Exception:
        return 0.0


def _get_model(
    model_name: str,
    prompt_wav_path: str,
    prompt_text: str,
    reference_wav_path: str | None,
):
    """Lazy-load and cache the Nano-vLLM-VoxCPM server (singleton per process).

    Matches the reference project's VoxCPMProvider.__init__ + _init_conditioning
    exactly: from_pretrained receives only model/devices/max_num_batched_tokens/
    max_num_seqs/gpu_memory_utilization (engine defaults for the rest), then
    add_prompt is tried first with encode_latents as fallback, then the
    optional reference clip is encoded to latents.

    The first call wins; later calls reuse the existing server.
    """
    global _model, _model_name, _prompt_id, _prompt_latents_fallback
    global _ref_audio_latents
    if _model is not None and _model_name == model_name:
        return _model

    from nanovllm_voxcpm import VoxCPM as NanoVoxCPM

    device = DEFAULT_DEVICE
    if device == "auto":
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_index = int(device.split(":")[-1]) if device.startswith("cuda:") else 0

    local_model_path = _resolve_local_model(model_name)

    print(f"[voxcpm] Loading model: {model_name} (device={device}) ...", flush=True)
    _model = NanoVoxCPM.from_pretrained(
        model=local_model_path,
        devices=[gpu_index],
        max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=DEFAULT_MAX_NUM_SEQS,
        gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    _model_name = model_name

    # Pre-encode the prompt pair once (add_prompt first, encode_latents fallback).
    prompt_bytes = _read_wav_bytes(prompt_wav_path)
    prompt_dur = _wav_duration_seconds(prompt_wav_path)
    print(f"[voxcpm] Encoding prompt ({prompt_dur:.1f}s) ...", flush=True)
    try:
        _prompt_id = _model.add_prompt(prompt_bytes, "wav", prompt_text)
        _prompt_latents_fallback = None
    except (AttributeError, NotImplementedError):
        _prompt_id = None
        _prompt_latents_fallback = _model.encode_latents(prompt_bytes, "wav")

    # Optional reinforce/reference clip -> ref_audio_latents.
    if reference_wav_path:
        ref_dur = _wav_duration_seconds(reference_wav_path)
        print(f"[voxcpm] Encoding reference ({ref_dur:.1f}s) ...", flush=True)
        try:
            ref_bytes = _read_wav_bytes(reference_wav_path)
            _ref_audio_latents = _model.encode_latents(ref_bytes, "wav")
        except Exception:
            _ref_audio_latents = None
    else:
        _ref_audio_latents = None

    print("[voxcpm] Model ready.", flush=True)
    atexit.register(teardown)
    return _model


def is_model_loaded() -> bool:
    """Whether the Nano-vLLM server is currently loaded in memory."""
    return _model is not None


def teardown() -> None:
    """Stop the Nano-vLLM server and release CUDA memory."""
    global _model, _model_name, _prompt_id, _prompt_latents_fallback
    global _ref_audio_latents
    if _model is not None:
        try:
            _model.stop()
        except Exception:
            pass
        _model = None
        _model_name = None
        _prompt_id = None
        _prompt_latents_fallback = None
        _ref_audio_latents = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Text splitting (preserved — segments long text for stable generation)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_audio(
    text: str,
    prompt_wav_path: str,
    prompt_text: str,
    output_path: str | os.PathLike,
    reference_wav_path: str | None = None,
    model_name: str = "openbmb/VoxCPM2",
    cfg_value: float = DEFAULT_CFG_VALUE,
    temperature: float = DEFAULT_TEMPERATURE,
    max_generate_length: int = DEFAULT_MAX_GENERATE_LENGTH,
) -> dict:
    """Synthesize speech with Hi-Fi Cloning and write a .wav file.

    prompt_wav_path + prompt_text provide the prompt pair (exact transcript)
    used for alignment/continuation. reference_wav_path provides a separate
    clip that reinforces the timbre; if None, no ref_audio_latents are passed.

    The Nano-v-LMM server.generate() is a generator yielding numpy audio
    chunks; we concatenate them per segment, then concatenate segments.

    Returns a dict with audio metadata:
      sample_rate, duration_seconds, file_size_bytes, model, mode, created_at
    """
    output_path = Path(output_path)
    prompt_wav_path = str(prompt_wav_path)
    reference_wav_path = str(reference_wav_path) if reference_wav_path else None

    model = _get_model(model_name, prompt_wav_path, prompt_text, reference_wav_path)

    segments = _split_text(text)
    if not segments:
        raise ValueError("text is empty or contains no speakable content")

    all_wavs: list[np.ndarray] = []
    from tqdm import tqdm

    for seg in tqdm(segments, desc="Generating", unit="seg"):
        kwargs: dict = dict(
            target_text=seg,
            cfg_value=cfg_value,
            temperature=temperature,
            max_generate_length=max_generate_length,
        )
        if _prompt_id is not None:
            kwargs["prompt_id"] = _prompt_id
        else:
            kwargs["prompt_latents"] = _prompt_latents_fallback
            kwargs["prompt_text"] = prompt_text
        if _ref_audio_latents is not None:
            kwargs["ref_audio_latents"] = _ref_audio_latents

        chunks = [
            np.asarray(chunk, dtype=np.float32)
            for chunk in model.generate(**kwargs)
        ]
        if not chunks:
            raise RuntimeError("VoxCPM produced no audio chunks for segment")
        wav = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        all_wavs.append(wav)

    full_wav = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), full_wav, SAMPLE_RATE)

    file_size = output_path.stat().st_size
    duration = len(full_wav) / float(SAMPLE_RATE)

    print(
        f"[voxcpm] Audio: {len(segments)} segment(s), {duration:.1f}s, "
        f"{SAMPLE_RATE}Hz, {file_size/1024:.0f}KB -> {output_path.name}",
        flush=True,
    )

    return {
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": round(duration, 3),
        "file_size_bytes": file_size,
        "model": model_name,
        "mode": "hifi_clone",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate speech via Nano-vLLM-VoxCPM Hi-Fi Cloning."
    )
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
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default {DEFAULT_TEMPERATURE})")
    parser.add_argument("--max-generate-length", type=int, default=DEFAULT_MAX_GENERATE_LENGTH,
                        help=f"Max generated audio tokens per segment (default {DEFAULT_MAX_GENERATE_LENGTH})")
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
            temperature=args.temperature,
            max_generate_length=args.max_generate_length,
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

    print(json.dumps({"audio_path": filename, "metadata": result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())