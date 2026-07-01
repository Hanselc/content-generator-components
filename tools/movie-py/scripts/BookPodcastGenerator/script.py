"""BookPodcastGenerator script.

Builds a multi-episode podcast from a book spec: groups chapters into
episodes honoring internal min/target duration thresholds (a chapter's
duration is the sum of its content audio segments), then for each episode
concatenates (in order) welcome -> each chapter's content segments ->
summary intro -> each chapter's summary segments -> farewell, with
inter-segment silence and a trailing silence. Emits per-episode WAV + MP4
and a run-level metadata.json with per-chapter time marks.

Each chapter supplies its content (and optional summary) as an ARRAY of
audio filenames, which are concatenated to form that chapter's audio. This
lets a chapter be produced as multiple TTS segments that are joined into one
continuous chapter audio.

The MP4 is a 3-phase static-frame video shared across episodes (only the
audio track and total duration differ per episode). The background is pure
black; on-screen text is faint dark-gray (#222222, no outline) rendered with
the vendored DejaVu fonts (./assets). Each text phase fades in then back out
to black over 1s; the black phase is plain black with no text:
    phase 1 (0 - INTRO_MESSAGE_SECONDS):     black + intro_message text (regular)
    phase 2 (INTRO_MESSAGE_SECONDS -         black + book_title (bold, -80px) +
             INTRO_MESSAGE_SECONDS + TITLE_SECONDS)   author (regular, +80px)
    phase 3 (after that, to episode end):    black (no text)
The intro phase is skipped when no `intro_message` is provided; the title
phase then starts at t=0. The episode's audio plays continuously throughout
all three phases, so the black phase lengthens or shortens to match each
episode's duration.

This is a script module under movie-py's scripts registry: it exposes
build(ctx) -> dict, PARAM_SCHEMA and META, and reuses the shared primitives
in make_video.py (silent_audio_clip, FPS, CODEC). Text rendering is
self-contained (vendored DejaVu fonts under ./assets) so the video style
(faint dark-gray #222222 text on pure black, with per-phase 1s fades) is
independent of make_video's caption/title-overlay primitives. It does NOT
call make_video.assemble() because that forces a <timestamp>.mp4 name and a
single-mp4 result; here we need per-episode podcast_NN.{wav,mp4} naming
plus an extra WAV + metadata.json, so the finalizer is inlined.

Spec (podcast.json-style, read from ctx.spec_path; all audio filenames
resolve against ctx.input_folder):
    {
      "book_title": "Don Quijote",                       // required
      "author": "Cervantes",                            // required
      "intro_message": "Bienvenidos al podcast",        // optional, book-level
      "welcome":        "welcome.wav",                  // optional, book-level
      "summary_intro":  "summary_intro.wav",            // optional, book-level
      "farewell":       "farewell.wav",                  // optional, book-level
      "chapters": [                                      // required, non-empty
        {"title": "Capítulo 1",
         "content": ["ch01a.wav", "ch01b.wav"],          // optional, array
         "summary": ["sum01.wav"]},                      // optional, array
        {"title": "Capítulo 2",
         "content": ["ch02.wav"],
         "summary": ["sum02a.wav", "sum02b.wav"]},
        {"title": "Capítulo 3",                          // summary-only chapter
         "content": [],
         "summary": ["sum03.wav"]}
      ]
    }

Required:
  - book_title, author, chapters (non-empty array).
Optional:
  - intro_message (book-level on-screen text for the video intro phase;
    omitted = skip the intro phase)
  - welcome, summary_intro, farewell (book-level audio; omitted = no such
    segment in any episode).

welcome / summary_intro / farewell are optional and book-level (one each,
played at the start / before summaries / at the end of every episode). Each
chapter's `content` and `summary` are optional arrays of audio filenames;
either or both may be omitted/empty. A chapter with neither content nor
summary audio is silently skipped (it has nothing to contribute to any
episode). A chapter with empty content but a non-empty summary is a valid
"summary-only" chapter: it contributes only its summary segment(s) to its
episode (no content section/time mark) and is included in episode grouping
using its content + summary total duration.

Episode grouping and silence timings (min/target chapter duration,
inter-segment silence, trailing silence) are script-internal constants and
are NOT configurable via the spec or params.

Params: PARAM_SCHEMA is empty (no overridable params). Any caller-supplied
param key is rejected by the schema.

Result: section-aware dict (one section per episode) per the movie-py
contract, with `metadata.episodes` carrying per-episode time_marks and
`outputs` listing every produced file (per-episode WAV + MP4 + metadata.json)
as {index, path, kind, label, section?} descriptors.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import make_video

PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}

META: dict[str, Any] = {
    "description": (
        "Multi-episode book podcast: groups chapters (each a concatenation of "
        "multiple content audio segments) into episodes honoring min/target "
        "duration, then concatenates welcome + contents + summary intro + "
        "summaries + farewell with silence padding. Emits per-episode WAV + "
        "MP4 (3-phase static-frame video: intro message -> title/author -> "
        "black, audio throughout; faint #222222 text on pure black with 1s "
        "fades, vendored DejaVu fonts) and a run-level metadata.json with "
        "per-chapter time marks."
    ),
    "version": "2.0.0",
}

# Defaults (used when neither spec nor params supply a value).
DEFAULT_MIN_CHAPTER_DURATION_S = 300.0
DEFAULT_TARGET_CHAPTER_DURATION_S = 600.0
DEFAULT_SILENCE_BETWEEN_MS = 1000
DEFAULT_TRAILING_SILENCE_MS = 5000

# Video render settings for the MP4 deliverable.
VIDEO_W = 1920
VIDEO_H = 1080
BG_COLOR = (0, 0, 0)  # pure black
# Text style (matches LibrosParaDormirMejor deliverables: faint dark-gray on
# black, no outline, regular DejaVu for author/intro, bold DejaVu for title).
TEXT_COLOR = (34, 34, 34)  # #222222, barely visible on black
TITLE_FONT_SIZE = 36
AUTHOR_FONT_SIZE = 24
INTRO_FONT_SIZE = 32
# Title sits 80px above the vertical center, author 80px below it (matches
# the reference project's (h-size)/2-80 / (h-size)/2+80 layout, so the two
# never overlap).
TITLE_Y_OFFSET = -80
AUTHOR_Y_OFFSET = 80
# Fonts are vendored under this script's assets/ folder so the video style is
# self-contained and independent of system-installed fonts.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
REGULAR_FONT_NAME = "DejaVuSans.ttf"
BOLD_FONT_NAME = "DejaVuSans-Bold.ttf"
# Per-phase fade-to/from-black duration in seconds (matches the reference).
FADE_SECONDS = 1.0
# 3-phase video timing (hardcoded). Phase 1 shows the optional intro_message
# text; phase 2 shows book_title + author; phase 3 is plain black. The
# episode audio plays throughout, so phase 3 stretches to match each
# episode's duration. Each text phase fades in then back out to black over
# FADE_SECONDS at its start and end.
INTRO_MESSAGE_SECONDS = 5.0
TITLE_SECONDS = 5.0
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


def _resolve_list(
    input_folder: Path,
    names: list[str] | None,
    field: str,
    chapter_idx: int,
) -> list[Path]:
    """Resolve a list of spec audio filenames against input_folder.

    `names` must be a non-empty list of non-empty strings (when provided).
    Every entry must resolve to an existing file; missing files raise a
    MovieSpecError naming the chapter index and field. Returns the list of
    resolved absolute Paths (in order). Empty/None input -> empty list.
    """
    if not names:
        return []
    if not isinstance(names, list):
        raise make_video.MovieSpecError(
            f"chapters[{chapter_idx}] {field} must be an array of filenames")
    paths: list[Path] = []
    for j, n in enumerate(names):
        if not isinstance(n, str) or not n:
            raise make_video.MovieSpecError(
                f"chapters[{chapter_idx}] {field}[{j}] must be a non-empty "
                f"string filename")
        p = (input_folder / n).resolve()
        if not p.is_file():
            raise make_video.MovieSpecError(
                f"chapters[{chapter_idx}] {field}[{j}] audio not found in "
                f"input_folder: {n!r}")
        paths.append(p)
    return paths


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
    """Group chapter indices into episodes.

    Grouping is based on each chapter's content + summary total duration
    (`_audio_duration`). Walk chapters in order; start a new group when
    adding the next chapter would exceed target_s. Afterwards merge undersized
    groups (< min_s): non-last groups merge forward; if the last group is
    undersized and there is more than one group, merge it backward. Chapters
    with no usable audio (neither content nor summary) are skipped; the
    caller already drops such chapters before this runs.

    Returns a list of groups, each a list of indices into `chapters`.
    """
    # (index, duration) for chapters that have any usable audio.
    usable: list[tuple[int, float]] = []
    for i, ch in enumerate(chapters):
        dur = ch.get("_audio_duration")
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


def _load_face(size: int, bold: bool = False) -> Any:
    """Load a PIL FreeTypeFont face from this script's assets/ folder.

    Falls back to the system DejaVu fonts and finally to PIL's default font if
    the vendored files are missing. `bold` selects DejaVuSans-Bold.ttf, else
    DejaVuSans.ttf.
    """
    from PIL import ImageFont
    primary = ASSETS_DIR / (BOLD_FONT_NAME if bold else REGULAR_FONT_NAME)
    fallback_name = BOLD_FONT_NAME if bold else REGULAR_FONT_NAME
    try:
        return ImageFont.truetype(str(primary), size)
    except Exception:
        try:
            return ImageFont.truetype(fallback_name, size)
        except Exception:
            try:
                return ImageFont.truetype("DejaVuSans.ttf", size)
            except Exception:
                return ImageFont.load_default()


def _render_text_layer(
    size: tuple[int, int],
    text: str,
    font_size: int,
    bold: bool,
    y_center_offset: int,
) -> Any:
    """Render a single line of faint dark-gray text on a transparent RGBA
    layer, horizontally centered, vertically centered +/- y_center_offset.

    No stroke/outline (matches the reference project's barely-visible-on-black
    look). The returned RGBA image has a fully transparent background so the
    caller can animate its alpha per-frame for fade in/out.
    """
    from PIL import Image, ImageDraw
    w, h = size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if not text:
        return layer
    draw = ImageDraw.Draw(layer)
    font = _load_face(font_size, bold=bold)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (w - tw) // 2 - bbox[0]
    ty = (h - th) // 2 - bbox[1] + y_center_offset
    draw.text((tx, ty), text, fill=TEXT_COLOR + (255,), font=font)
    return layer


def _build_black_frame() -> np.ndarray:
    """Build a single 1920x1080 pure-black RGB frame (no text)."""
    from PIL import Image
    return np.array(Image.new("RGB", (VIDEO_W, VIDEO_H), BG_COLOR))


def _build_intro_layer(message: str) -> Any:
    """Build a transparent RGBA layer with the intro_message centered (regular
    DejaVu, size INTRO_FONT_SIZE). Composited by the MP4 writer with a
    time-varying alpha during the intro phase."""
    return _render_text_layer(
        (VIDEO_W, VIDEO_H), message,
        font_size=INTRO_FONT_SIZE, bold=False, y_center_offset=0,
    )


def _build_title_layer(title: str) -> Any:
    """Build a transparent RGBA layer with the book title (bold DejaVu, size
    TITLE_FONT_SIZE, 80px above center)."""
    return _render_text_layer(
        (VIDEO_W, VIDEO_H), title,
        font_size=TITLE_FONT_SIZE, bold=True, y_center_offset=TITLE_Y_OFFSET,
    )


def _build_author_layer(author: str) -> Any:
    """Build a transparent RGBA layer with the author (regular DejaVu, size
    AUTHOR_FONT_SIZE, 80px below center)."""
    return _render_text_layer(
        (VIDEO_W, VIDEO_H), author,
        font_size=AUTHOR_FONT_SIZE, bold=False, y_center_offset=AUTHOR_Y_OFFSET,
    )


def _write_wav(audio_clip, out_path: Path) -> None:
    """Write an audio clip as a 16-bit PCM WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_clip.write_audiofile(
        str(out_path),
        codec="pcm_s16le",
        ffmpeg_params=["-ac", "2"],
        logger=None,
    )


def _phase_alpha(t: float, start_s: float, end_s: float,
                 fade_s: float) -> float:
    """Per-phase fade-in/fade-out alpha, clamped to [0, 1].

    Matches the reference project's drawtext alpha expression: ramps 0->1 over
    fade_s at the phase start, holds 1, ramps 1->0 over fade_s at the phase
    end. Returns 0 outside [start_s, end_s].
    """
    if t < start_s or t >= end_s:
        return 0.0
    fd = min(fade_s, (end_s - start_s) / 2.0)
    local = t - start_s
    phase_len = end_s - start_s
    if local < fd:
        return local / fd if fd > 0 else 1.0
    if local > phase_len - fd:
        return (phase_len - local) / fd if fd > 0 else 1.0
    return 1.0


def _write_mp4(phases: list[dict],
               layers: dict[str, Any],
               black_frame: np.ndarray,
               audio_clip, duration: float,
               out_path: Path) -> None:
    """Write a static-phase MP4 (1920x1080) with the given audio.

    `phases` is a list of {kind, start_s, end_s} in timeline order; `kind` is
    one of "intro", "title", "black". `layers` maps "intro" -> RGBA layer,
    "title" -> RGBA layer, "author" -> RGBA layer (the title phase composites
    both title and author layers at the shared phase alpha). The black phase
    has no layer. Each text phase fades in/out to black over FADE_SECONDS.
    Audio plays across the whole `duration`.
    """
    from PIL import Image
    from moviepy import VideoClip
    out_path.parent.mkdir(parents=True, exist_ok=True)

    black_rgba = Image.fromarray(black_frame).convert("RGBA")
    h, w = black_frame.shape[:2]

    # Pre-bind each phase's kind to speed up the per-frame lookup.
    def make_frame(t: float):
        active = None
        for ph in phases:
            if ph["start_s"] <= t < ph["end_s"]:
                active = ph
                break
        if active is None:
            active = phases[-1]
        kind = active["kind"]
        if kind == "black":
            return black_frame
        alpha = _phase_alpha(t, active["start_s"], active["end_s"],
                             FADE_SECONDS)
        if alpha <= 0.0:
            return black_frame
        frame = black_rgba
        if kind == "intro":
            frame = _composite_alpha(frame, layers["intro"], alpha)
        elif kind == "title":
            frame = _composite_alpha(frame, layers["title"], alpha)
            frame = _composite_alpha(frame, layers["author"], alpha)
        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration).with_audio(audio_clip)
    clip.write_videofile(
        str(out_path),
        fps=make_video.FPS,
        codec=make_video.CODEC,
        audio_codec="aac",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
        logger=None,
    )
    clip.close()


def _composite_alpha(base_rgba: Any, layer_rgba: Any, alpha: float) -> Any:
    """Composite `layer_rgba` onto `base_rgba` with a global scalar alpha.

    Both are RGBA PIL Images. The layer's existing per-pixel alpha is scaled by
    `alpha` (so text stays crisp while the whole text fades together). Returns
    a new RGBA PIL Image.
    """
    if alpha <= 0:
        return base_rgba
    from PIL import Image
    if alpha >= 1.0:
        return Image.alpha_composite(base_rgba, layer_rgba)
    scaled = layer_rgba.copy()
    a = scaled.split()[3]
    scaled.putalpha(a.point(lambda v: int(v * alpha)))
    return Image.alpha_composite(base_rgba, scaled)


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
    intro_message = raw.get("intro_message")
    if intro_message is not None:
        if not isinstance(intro_message, str) or not intro_message.strip():
            raise make_video.MovieSpecError(
                "intro_message must be a non-empty string if present")
        intro_message = intro_message.strip()

    # Episode grouping / silence timings are script-internal defaults; they
    # are NOT overridable via the spec or params (PARAM_SCHEMA is empty).
    min_s = DEFAULT_MIN_CHAPTER_DURATION_S
    target_s = DEFAULT_TARGET_CHAPTER_DURATION_S
    silence_ms = DEFAULT_SILENCE_BETWEEN_MS
    trailing_ms = DEFAULT_TRAILING_SILENCE_MS

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

    # Resolve per-chapter audio and probe durations.
    # `content` and `summary` are each optional arrays; a chapter with neither
    # is skipped (nothing to contribute to any episode). A chapter with empty
    # content but a non-empty summary is a valid "summary-only" chapter.
    chapters: list[dict] = []
    skipped_chapters: list[int] = []
    for i, ch in enumerate(chapters_raw):
        if not isinstance(ch, dict):
            raise make_video.MovieSpecError(f"chapters[{i}] must be an object")
        title = ch.get("title") or f"Chapter {i + 1}"
        content_paths = _resolve_list(input_folder, ch.get("content"),
                                       "content", i)
        summary_paths = _resolve_list(input_folder, ch.get("summary"),
                                       "summary", i)
        if not content_paths and not summary_paths:
            # Nothing to contribute; skip silently.
            skipped_chapters.append(i)
            continue
        content_duration = sum(_audio_duration(p) for p in content_paths)
        summary_duration = sum(_audio_duration(p) for p in summary_paths)
        chapters.append({
            "title": title,
            "content_paths": content_paths,
            "summary_paths": summary_paths,
            "_content_duration": content_duration,
            # Content + summary total drives episode grouping so that
            # summary-only chapters still get their own grouping weight.
            "_audio_duration": content_duration + summary_duration,
        })

    if skipped_chapters:
        print(f"[BookPodcastGenerator] skipped {len(skipped_chapters)} "
              f"chapter(s) with no content or summary audio: "
              f"{skipped_chapters}", flush=True)

    # --- Group chapters into episodes -------------------------------------
    groups = _group_chapters(chapters, target_s=target_s, min_s=min_s)
    if not groups:
        raise make_video.MovieSpecError(
            "no usable chapter audio (content or summary) to build episodes "
            "from")

    output_folder = Path(ctx.output_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    silence_s = silence_ms / 1000.0
    trailing_s = trailing_ms / 1000.0

    # Static frames / text layers for the 3-phase video, pre-rendered once and
    # shared across episodes (only audio + total duration differ per episode).
    # The MP4 writer composites the RGBA text layers onto a pure-black frame
    # with a per-phase fade in/out alpha, so no fades are baked into pixels.
    black_frame = _build_black_frame()
    title_layer = _build_title_layer(book_title)
    author_layer = _build_author_layer(author)
    intro_layer = (_build_intro_layer(intro_message)
                   if intro_message else None)
    layers = {
        "intro": intro_layer,
        "title": title_layer,
        "author": author_layer,
    }

    # Build the phase descriptor list. The intro phase is omitted when there
    # is no intro_message; the title phase then starts at t=0. The final black
    # phase runs to +inf so it stretches to any episode duration.
    if intro_layer is not None:
        video_phases: list[dict] = [
            {"kind": "intro", "start_s": 0.0,
             "end_s": INTRO_MESSAGE_SECONDS},
            {"kind": "title", "start_s": INTRO_MESSAGE_SECONDS,
             "end_s": INTRO_MESSAGE_SECONDS + TITLE_SECONDS},
            {"kind": "black",
             "start_s": INTRO_MESSAGE_SECONDS + TITLE_SECONDS,
             "end_s": float("inf")},
        ]
    else:
        video_phases = [
            {"kind": "title", "start_s": 0.0, "end_s": TITLE_SECONDS},
            {"kind": "black", "start_s": TITLE_SECONDS,
             "end_s": float("inf")},
        ]

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

        # Build the ordered audio segment list. Each entry is either a
        # book-level single clip (welcome/summary_intro/farewell) or a
        # chapter, which contributes its content_paths / summary_paths (an
        # array of clips concatenated in order). The `title` is used for the
        # time mark; a chapter gets ONE mark at its first content clip.
        # `kind`/`paths` drive which clips are appended.
        segs: list[dict] = []
        if welcome_path:
            segs.append({"title": WELCOME_TITLE, "kind": "book",
                         "paths": [welcome_path]})
        for ci in group:
            segs.append({"title": chapters[ci]["title"], "kind": "content",
                         "paths": chapters[ci]["content_paths"], "ci": ci})
        if summary_intro_path:
            segs.append({"title": SUMMARY_INTRO_TITLE, "kind": "book",
                         "paths": [summary_intro_path]})
        for ci in group:
            sp = chapters[ci]["summary_paths"]
            if sp:
                segs.append({"title": chapters[ci]["title"], "kind": "summary",
                             "paths": sp, "ci": ci})
        if farewell_path:
            segs.append({"title": FAREWELL_TITLE, "kind": "book",
                         "paths": [farewell_path]})

        # Concatenate with per-section time marks + inter-segment silence
        # after every audio clip, and trailing silence at the very end.
        time_marks: list[dict] = []
        clips: list = []
        cursor = 0.0  # seconds within this episode
        for seg in segs:
            paths = seg["paths"]
            if not paths:
                continue
            # One time mark per section (per book-level clip / per chapter
            # content / per chapter summary), at its first audio clip's start.
            time_marks.append({
                "title": seg["title"],
                "seconds": round(cursor, 1),
                "time": _fmt_time(cursor),
            })
            for p in paths:
                a = AudioFileClip(str(p))
                clips_to_close.append(a)
                clips.append(a)
                cursor += float(a.duration)
                # Inter-segment silence after every audio clip.
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

        # --- Write MP4 (3-phase static frames + AAC audio) ----------------
        _write_mp4(video_phases, layers, black_frame,
                   episode_audio, episode_duration, ep_mp4)
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
        "intro_message": intro_message,
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
            "intro_message": intro_message,
            "episode_count": len(episodes_meta),
            "episodes": episodes_meta,
            "metadata_file": str(output_folder / "metadata.json"),
        },
    }