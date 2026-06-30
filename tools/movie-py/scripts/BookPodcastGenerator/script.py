"""BookPodcastGenerator script.

Builds a multi-episode podcast from a book spec: groups full-chapter audio
files into episodes honoring min/target duration, then for each episode
concatenates (in order) welcome -> chapter contents -> summary intro ->
chapter summaries -> farewell, with inter-segment silence and a trailing
silence. Emits per-episode WAV + MP4 (black background, faded title/author
overlay, AAC audio) and a run-level metadata.json with per-segment time marks.

This is a script module under movie-py's scripts registry: it exposes
build(ctx) -> dict, PARAM_SCHEMA and META, and reuses the shared primitives
in make_video.py (silent_audio_clip, render_title_overlay,
overlay_caption_on_array, load_font, FPS, CODEC). It does NOT call
make_video.assemble() because that forces a <timestamp>.mp4 name and a
single-mp4 result; here we need per-episode podcast_NN.{wav,mp4} naming
plus an extra WAV + metadata.json, so the finalizer is inlined.

Spec (podcast.json-style, read from ctx.spec_path; all audio filenames
resolve against ctx.input_folder):
    {
      "book_title": "Don Quijote",                       // required
      "author": "Cervantes",                            // required
      "min_chapter_duration_seconds": 300,              // optional, default 300
      "target_chapter_duration_seconds": 600,           // optional, default 600
      "silence_between_segments_ms": 1000,              // optional, default 1000
      "trailing_silence_ms": 5000,                       // optional, default 5000
      "welcome":        "welcome.wav",                  // optional, book-level
      "summary_intro":  "summary_intro.wav",            // optional, book-level
      "farewell":       "farewell.wav",                  // optional, book-level
      "chapters": [                                      // required, non-empty
        {"title": "Capítulo 1", "content": "ch01.wav", "summary": "sum01.wav"},
        {"title": "Capítulo 2", "content": "ch02.wav", "summary": "sum02.wav"}
      ]
    }

Required:
  - book_title, author, chapters (non-empty array).
Optional (with built-in defaults):
  - min_chapter_duration_seconds (default 300)
  - target_chapter_duration_seconds (default 600)
  - silence_between_segments_ms (default 1000)
  - trailing_silence_ms (default 5000)
  - welcome, summary_intro, farewell (book-level audio; omitted = no such
    segment in any episode).

welcome / summary_intro / farewell are optional and book-level (one each,
played at the start / before summaries / at the end of every episode). Each
chapter requires `content`; `summary` is optional per chapter.

Params (validated against PARAM_SCHEMA, override spec values; all optional
and default to the spec value or the built-in default above):
    min_chapter_duration_seconds, target_chapter_duration_seconds,
    silence_between_segments_ms, trailing_silence_ms  (all numbers >= 0).

Result: section-aware dict (one section per episode) per the movie-py
contract, with `metadata.episodes` carrying per-episode time_marks and
`outputs` listing every produced file (per-episode WAV + MP4 + metadata.json)
as {index, path, kind, label, section?} descriptors.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import make_video

PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "min_chapter_duration_seconds": {
            "type": "number",
            "minimum": 0,
            "description": "Minimum total content duration per episode; "
                           "undersized groups are merged.",
        },
        "target_chapter_duration_seconds": {
            "type": "number",
            "minimum": 0,
            "description": "Soft threshold: a new episode starts when adding "
                           "the next chapter would exceed this.",
        },
        "silence_between_segments_ms": {
            "type": "number",
            "minimum": 0,
            "description": "Silence inserted after every segment within an "
                           "episode (ms).",
        },
        "trailing_silence_ms": {
            "type": "number",
            "minimum": 0,
            "description": "Silence appended once at the very end of each "
                           "episode (ms).",
        },
    },
}

META: dict[str, Any] = {
    "description": (
        "Multi-episode book podcast: groups full-chapter audio into episodes "
        "honoring min/target duration, then concatenates welcome + contents "
        "+ summary intro + summaries + farewell with silence padding. "
        "Emits per-episode WAV + MP4 and a run-level metadata.json with "
        "per-segment time marks."
    ),
    "version": "1.0.0",
}

# Defaults (used when neither spec nor params supply a value).
DEFAULT_MIN_CHAPTER_DURATION_S = 300.0
DEFAULT_TARGET_CHAPTER_DURATION_S = 600.0
DEFAULT_SILENCE_BETWEEN_MS = 1000
DEFAULT_TRAILING_SILENCE_MS = 5000

# Video render settings for the MP4 deliverable.
VIDEO_W = 1920
VIDEO_H = 1080
BG_COLOR = (8, 8, 12)  # near-black
WELCOME_TITLE = "Introducción"
SUMMARY_INTRO_TITLE = "Resumen"
FAREWELL_TITLE = "Despedida"
EPISODE_TITLE_TEMPLATE = "Podcast Chapter {i}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_time(seconds: float) -> str:
    """Format seconds as M:SS (minutes as int, seconds zero-padded to 2)."""
    m = int(seconds // 60)
    s = int(round(seconds - 60 * m))
    if s == 60:  # rounding edge (e.g. 59.999 -> 60.0)
        m += 1
        s = 0
    return f"{m}:{s:02d}"


def _resolve(input_folder: Path, name: str | None) -> Path | None:
    """Resolve a spec audio filename against input_folder; None if absent."""
    if not name:
        return None
    p = (input_folder / name).resolve()
    return p if p.is_file() else None


def _audio_duration(path: Path) -> float:
    """Probe an audio file's duration in seconds via moviepy."""
    from moviepy import AudioFileClip
    with AudioFileClip(str(path)) as a:
        return float(a.duration)


def _group_chapters(
    chapters: list[dict],
    target_s: float,
    min_s: float,
) -> list[list[int]]:
    """Group content chapter indices into episodes.

    Grouping is based on content durations only. Walk chapters in order; start
    a new group when adding the next chapter would exceed target_s. Afterwards
    merge undersized groups (< min_s): non-last groups merge forward; if the
    last group is undersized and there is more than one group, merge it
    backward. Chapters with no usable content audio are skipped.

    Returns a list of groups, each a list of original chapter indices.
    """
    # (index, duration) for chapters that have a content file.
    usable: list[tuple[int, float]] = []
    for i, ch in enumerate(chapters):
        dur = ch.get("_content_duration")
        if dur is None or dur <= 0:
            continue
        usable.append((i, dur))

    if not usable:
        return []

    # Initial greedy grouping by target.
    groups: list[list[int]] = []
    cur: list[int] = []
    cur_dur = 0.0
    for idx, dur in usable:
        if cur and (cur_dur + dur) > target_s:
            groups.append(cur)
            cur = []
            cur_dur = 0.0
        cur.append(idx)
        cur_dur += dur
    if cur:
        groups.append(cur)

    # Merge undersized groups (non-last forward, last backward).
    def group_dur(g: list[int]) -> float:
        return sum(next(d for (i2, d) in usable if i2 == i) for i in g)

    changed = True
    while changed and len(groups) > 1:
        changed = False
        for gi in range(len(groups) - 1):
            if group_dur(groups[gi]) < min_s:
                groups[gi + 1] = groups[gi] + groups[gi + 1]
                del groups[gi]
                changed = True
                break
        if changed:
            continue
        if len(groups) > 1 and group_dur(groups[-1]) < min_s:
            groups[-2] = groups[-2] + groups[-1]
            del groups[-1]
            changed = True

    return groups


def _build_black_title_frame(text_lines: list[str]) -> np.ndarray:
    """Build a single 1920x1080 near-black RGB frame with centered, faded
    title/author text using make_video's caption primitives."""
    from PIL import Image
    h, w = VIDEO_H, VIDEO_W
    base = Image.new("RGB", (w, h), BG_COLOR)
    arr = np.array(base)
    for line in text_lines:
        if not line:
            continue
        arr = make_video.overlay_caption_on_array(arr, line)
    return arr


def _write_wav(audio_clip, out_path: Path) -> None:
    """Write an audio clip as a 16-bit PCM WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_clip.write_audiofile(
        str(out_path),
        codec="pcm_s16le",
        ffmpeg_params=["-ac", "2"],
        logger=None,
    )


def _write_mp4(frame_arr: np.ndarray, audio_clip, duration: float,
               out_path: Path) -> None:
    """Write a static-frame MP4 (1920x1080) with the given audio."""
    from moviepy import ImageClip
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip = ImageClip(frame_arr).with_duration(duration).with_audio(audio_clip)
    clip.write_videofile(
        str(out_path),
        fps=make_video.FPS,
        codec=make_video.CODEC,
        audio_codec="aac",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
        logger=None,
    )
    clip.close()


def build(ctx: SimpleNamespace) -> dict:
    """Build the book podcast. See module docstring for the contract."""
    from moviepy import AudioFileClip, concatenate_audioclips

    params: dict = ctx.params or {}

    # --- Load spec --------------------------------------------------------
    try:
        raw = json.loads(Path(ctx.spec_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise make_video.MovieSpecError(f"spec is not valid JSON: {e}")
    if not isinstance(raw, dict):
        raise make_video.MovieSpecError("spec root must be an object")

    book_title = raw.get("book_title") or "Untitled"
    author = raw.get("author") or "Unknown"

    def num_opt(key: str, default: float, spec_key: str | None = None) -> float:
        spec_key = spec_key or key
        v = raw.get(spec_key, params.get(key, default))
        try:
            return float(v)
        except (TypeError, ValueError):
            raise make_video.MovieSpecError(f"{spec_key} must be a number")

    min_s = num_opt("min_chapter_duration_seconds", DEFAULT_MIN_CHAPTER_DURATION_S)
    target_s = num_opt("target_chapter_duration_seconds",
                       DEFAULT_TARGET_CHAPTER_DURATION_S)
    silence_ms = num_opt("silence_between_segments_ms", DEFAULT_SILENCE_BETWEEN_MS)
    trailing_ms = num_opt("trailing_silence_ms", DEFAULT_TRAILING_SILENCE_MS)

    if min_s < 0 or target_s < 0 or silence_ms < 0 or trailing_ms < 0:
        raise make_video.MovieSpecError("duration/silence values must be >= 0")
    if target_s < min_s:
        raise make_video.MovieSpecError(
            "target_chapter_duration_seconds must be >= "
            "min_chapter_duration_seconds")

    chapters_raw = raw.get("chapters")
    if not isinstance(chapters_raw, list) or not chapters_raw:
        raise make_video.MovieSpecError("chapters must be a non-empty array")

    input_folder = Path(ctx.input_folder).resolve()

    # Resolve book-level audio segments.
    welcome_path = _resolve(input_folder, raw.get("welcome"))
    summary_intro_path = _resolve(input_folder, raw.get("summary_intro"))
    farewell_path = _resolve(input_folder, raw.get("farewell"))
    if raw.get("welcome") and not welcome_path:
        raise make_video.MovieSpecError(
            f"welcome audio not found in input_folder: {raw.get('welcome')!r}")
    if raw.get("summary_intro") and not summary_intro_path:
        raise make_video.MovieSpecError(
            f"summary_intro audio not found in input_folder: "
            f"{raw.get('summary_intro')!r}")
    if raw.get("farewell") and not farewell_path:
        raise make_video.MovieSpecError(
            f"farewell audio not found in input_folder: {raw.get('farewell')!r}")

    # Resolve per-chapter audio and probe content durations.
    chapters: list[dict] = []
    for i, ch in enumerate(chapters_raw):
        if not isinstance(ch, dict):
            raise make_video.MovieSpecError(f"chapters[{i}] must be an object")
        title = ch.get("title") or f"Chapter {i + 1}"
        content_path = _resolve(input_folder, ch.get("content"))
        if not content_path:
            raise make_video.MovieSpecError(
                f"chapters[{i}] content audio not found in input_folder: "
                f"{ch.get('content')!r}")
        summary_path = _resolve(input_folder, ch.get("summary"))
        if ch.get("summary") and not summary_path:
            raise make_video.MovieSpecError(
                f"chapters[{i}] summary audio not found in input_folder: "
                f"{ch.get('summary')!r}")
        chapters.append({
            "title": title,
            "content_path": content_path,
            "summary_path": summary_path,
            "_content_duration": _audio_duration(content_path),
        })

    # --- Group chapters into episodes -------------------------------------
    groups = _group_chapters(chapters, target_s=target_s, min_s=min_s)
    if not groups:
        raise make_video.MovieSpecError(
            "no usable chapter content audio to build episodes from")

    output_folder = Path(ctx.output_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    silence_s = silence_ms / 1000.0
    trailing_s = trailing_ms / 1000.0

    # Static title frame reused for every episode's MP4.
    title_frame = _build_black_title_frame([book_title, author])

    episodes_meta: list[dict] = []
    sections: list[dict] = []
    outputs: list[dict] = []
    clips_to_close: list = []
    timeline_cursor = 0.0  # across all episodes, for sections
    started_at = ctx.common.get("started_at") or _now_iso()
    total_duration = 0.0

    for ep_idx, group in enumerate(groups, start=1):
        ep_started = _now_iso()
        ep_wav = output_folder / f"podcast_{ep_idx:02d}.wav"
        ep_mp4 = output_folder / f"podcast_{ep_idx:02d}.mp4"
        ep_title = EPISODE_TITLE_TEMPLATE.format(i=ep_idx)

        # Build the ordered segment list: (title, path) ; path None = skip.
        segs: list[tuple[str, Path | None]] = []
        if welcome_path:
            segs.append((WELCOME_TITLE, welcome_path))
        for ci in group:
            segs.append((chapters[ci]["title"], chapters[ci]["content_path"]))
        if summary_intro_path:
            segs.append((SUMMARY_INTRO_TITLE, summary_intro_path))
        for ci in group:
            sp = chapters[ci]["summary_path"]
            if sp:
                segs.append((chapters[ci]["title"] + ".", sp))
        if farewell_path:
            segs.append((FAREWELL_TITLE, farewell_path))

        # Concatenate with time marks + inter-segment + trailing silence.
        time_marks: list[dict] = []
        clips: list = []
        cursor = 0.0  # seconds within this episode
        for s_idx, (seg_title, seg_path) in enumerate(segs):
            if seg_path is None:
                continue
            time_marks.append({
                "title": seg_title,
                "seconds": round(cursor, 1),
                "time": _fmt_time(cursor),
            })
            a = AudioFileClip(str(seg_path))
            clips_to_close.append(a)
            clips.append(a)
            cursor += float(a.duration)
            # Inter-segment silence after every segment.
            if silence_s > 0:
                sil = make_video.silent_audio_clip(silence_s, fps=44100,
                                                   nchannels=2)
                clips.append(sil)
                cursor += silence_s
        # Trailing silence at the very end of the episode.
        if trailing_s > 0:
            sil = make_video.silent_audio_clip(trailing_s, fps=44100,
                                                nchannels=2)
            clips.append(sil)
            cursor += trailing_s

        episode_audio = concatenate_audioclips(clips)
        episode_duration = float(episode_audio.duration)

        # --- Write WAV ----------------------------------------------------
        _write_wav(episode_audio, ep_wav)
        print(f"[BookPodcastGenerator] episode {ep_idx}: "
              f"{episode_duration:.1f}s, {len(group)} chapters -> "
              f"{ep_wav.name}", flush=True)

        # --- Write MP4 (static title frame + AAC audio) -------------------
        _write_mp4(title_frame, episode_audio, episode_duration, ep_mp4)
        print(f"[BookPodcastGenerator] episode {ep_idx}: MP4 -> {ep_mp4.name}",
              flush=True)

        episode_audio.close()

        ep_finished = _now_iso()
        episodes_meta.append({
            "index": ep_idx,
            "audio_file": ep_wav.name,
            "video_file": ep_mp4.name,
            "title": ep_title,
            "source_chapters": list(group),
            "duration_seconds": round(episode_duration, 2),
            "chapter_count": len(group),
            "time_marks": time_marks,
        })
        sections.append({
            "name": f"episode_{ep_idx}",
            "start_seconds": round(timeline_cursor, 3),
            "end_seconds": round(timeline_cursor + episode_duration, 3),
            "duration_seconds": round(episode_duration, 3),
            "started_at": ep_started,
            "finished_at": ep_finished,
        })
        section_name = f"episode_{ep_idx}"
        outputs.append({
            "index": len(outputs),
            "path": str(ep_wav),
            "kind": "audio",
            "label": f"{section_name}_audio",
            "section": section_name,
        })
        outputs.append({
            "index": len(outputs),
            "path": str(ep_mp4),
            "kind": "video",
            "label": f"{section_name}_video",
            "section": section_name,
        })
        timeline_cursor += episode_duration
        total_duration += episode_duration

    # --- Write run-level metadata.json ------------------------------------
    finished_at = _now_iso()
    metadata = {
        "book_title": book_title,
        "author": author,
        "started_at": started_at,
        "finished_at": finished_at,
        "episodes": episodes_meta,
    }
    (output_folder / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs.append({
        "index": len(outputs),
        "path": str(output_folder / "metadata.json"),
        "kind": "metadata",
        "label": "metadata",
    })

    # Close all opened AudioFileClips.
    for c in clips_to_close:
        try:
            c.close()
        except Exception:
            pass

    return {
        "script_id": ctx.common.get("script_id", ""),
        "spec_path": str(ctx.spec_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "total_duration_seconds": round(total_duration, 3),
        "sections": sections,
        "outputs": outputs,
        "metadata": {
            "book_title": book_title,
            "author": author,
            "episode_count": len(episodes_meta),
            "episodes": episodes_meta,
            "metadata_file": str(output_folder / "metadata.json"),
        },
    }