"""SocialMediaGenerator script.

Builds a social-media-style video: an opening composition (3 balanced Voronoi
shards with Ken Burns animation + title overlay) followed by a captioned
slideshow of the remaining images, with silence-padded audio and crossfades.

This reproduces the original movie-py build_video() behavior, but expressed as
a script module that composes the shared primitives from make_video.py and
finishes via make_video.assemble().

Spec (movie.json-style, read from ctx.spec_path):
    {
      "title": {"text": "...", "audio": "intro.mp3"} | "...",
      "images": [
        {"image": "0_xxx.png", "text": "caption", "audio": "0.mp3"},
        ...
      ]
    }

Image/audio filenames inside the spec resolve against ctx.input_folder.

Params (validated against PARAM_SCHEMA):
    display_seconds:     seconds each slide displays when it has no audio
                         (default 5)
    transition_seconds:  crossfade duration in seconds (default 1)

Result: section-aware dict (see make_video.assemble) with two sections:
    intro  -- opening composition
    slides -- captioned image slideshow
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

import make_video

PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "display_seconds": {
            "type": "number",
            "minimum": 0.1,
            "default": make_video.DEFAULT_DISPLAY,
            "description": "Seconds each slide displays when it has no audio.",
        },
        "transition_seconds": {
            "type": "number",
            "minimum": 0,
            "default": make_video.DEFAULT_TRANSITION,
            "description": "Crossfade duration in seconds.",
        },
    },
}

META: dict[str, Any] = {
    "description": (
        "Social-media-style video: opening Voronoi/Ken Burns composition with "
        "title overlay, followed by a captioned image slideshow with "
        "silence-padded audio crossfades."
    ),
    "version": "1.0.0",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build(ctx: SimpleNamespace) -> dict:
    """Build the social-media video. See module docstring for the contract."""
    from moviepy import ImageClip, AudioFileClip
    try:
        from moviepy.video.fx import CrossFadeIn
    except Exception:
        from moviepy.video.fx.all import CrossFadeIn  # type: ignore

    primitives = ctx.primitives
    params: dict = ctx.params or {}
    display_seconds = float(params.get("display_seconds", make_video.DEFAULT_DISPLAY))
    transition_seconds = float(params.get("transition_seconds", make_video.DEFAULT_TRANSITION))

    started_at = ctx.common.get("started_at") or _now_iso()

    # --- Load spec ---------------------------------------------------------
    title, entries = primitives.load_movie(ctx.spec_path, ctx.input_folder)
    image_paths = [e["path"] for e in entries]

    # Target size from the first image.
    with Image.open(image_paths[0]) as im:
        target_w, target_h = im.size
    target_size = (target_w, target_h)

    rng = np.random.default_rng(1234)

    clips_to_close: list = []
    composite_clips: list = []
    sections: list[dict] = []

    # --- Section 1: Intro composition -------------------------------------
    intro_section_started = _now_iso()
    intro_audio_clips: list = []
    if title.get("audio") is not None:
        title_audio = AudioFileClip(str(title["audio"]))
        intro_audio_clips.append(title_audio)
        intro_padded_audio = primitives.padded_audio(title_audio, transition_seconds)
        intro_duration = float(intro_padded_audio.duration)
        print(f"[SocialMediaGenerator] Building intro composition "
              f"({intro_duration:.1f}s, title audio {title_audio.duration:.1f}s) ...",
              flush=True)
    else:
        intro_padded_audio = None
        intro_duration = float(make_video.INTRO_SECONDS)
        print(f"[SocialMediaGenerator] Building intro composition "
              f"({intro_duration:.1f}s, silent) ...", flush=True)

    intro_clip = primitives.build_composition_clip(
        image_paths, title, target_w, target_h, intro_duration, rng)
    if intro_padded_audio is not None:
        intro_clip = intro_clip.with_audio(intro_padded_audio)

    intro_clip = intro_clip.with_start(0)
    composite_clips.append(intro_clip)
    clips_to_close.extend(intro_audio_clips)
    clips_to_close.append(intro_clip)
    intro_section_finished = _now_iso()
    sections.append({
        "name": "intro",
        "start_seconds": 0.0,
        "end_seconds": round(intro_duration, 3),
        "duration_seconds": round(intro_duration, 3),
        "started_at": intro_section_started,
        "finished_at": intro_section_finished,
    })

    # --- Section 2: Captioned image slideshow -----------------------------
    slides_section_started = _now_iso()
    image_clips: list = []
    durations: list[float] = []
    audio_clips: list = []
    pad = float(transition_seconds)
    total_entries = len(entries)
    for idx, e in enumerate(entries, start=1):
        print(f"[SocialMediaGenerator] Preparing slide {idx}/{total_entries} ...",
              flush=True)
        arr = primitives.load_normalized(e["path"], target_w, target_h)
        text = e.get("text")
        if text:
            arr = primitives.overlay_caption_on_array(arr, text)
        if e.get("audio") is not None:
            a = AudioFileClip(str(e["audio"]))
            audio_clips.append(a)
            padded = primitives.padded_audio(a, pad)
            dur = float(padded.duration)
            clip = ImageClip(arr).with_duration(dur).with_audio(padded)
        else:
            dur = float(display_seconds) + 2 * pad
            clip = ImageClip(arr).with_duration(dur)
        image_clips.append(clip)
        durations.append(dur)

    # Place image clips on the timeline with crossfades.
    # First image starts at (intro_duration - transition) so it crossfades over
    # the tail of the intro; subsequent images step by
    # (prev_duration - transition_seconds). Because each clip's duration
    # includes 2*transition of silent padding around its audio, the next clip's
    # audio starts exactly transition_seconds after the previous clip's audio
    # ends -- no audio overlap, and the visual crossfade happens entirely within
    # the silent tails.
    start = intro_duration - transition_seconds
    for i, clip in enumerate(image_clips):
        clip = clip.with_start(start)
        clip = clip.with_effects([CrossFadeIn(transition_seconds)])
        image_clips[i] = clip
        composite_clips.append(clip)
        start += durations[i] - transition_seconds

    clips_to_close.extend(image_clips)
    clips_to_close.extend(audio_clips)

    # After the loop, `start` is one transition past the last clip's end (it
    # was incremented after the last clip was placed). Add it back.
    total_duration = start + transition_seconds

    slides_section_finished = _now_iso()
    slides_start_on_timeline = intro_duration - transition_seconds
    sections.append({
        "name": "slides",
        "start_seconds": round(slides_start_on_timeline, 3),
        "end_seconds": round(total_duration, 3),
        "duration_seconds": round(total_duration - slides_start_on_timeline, 3),
        "started_at": slides_section_started,
        "finished_at": slides_section_finished,
    })

    # --- Assemble + write --------------------------------------------------
    extra_metadata = {
        "image_count": len(image_clips),
        "intro_seconds": round(intro_duration, 3),
        "title": title["text"],
        "title_audio": title["audio"].name if title.get("audio") else None,
        "transition_seconds": transition_seconds,
        "display_seconds": display_seconds,
    }

    return primitives.assemble(
        composite_clips=composite_clips,
        sections=sections,
        output_folder=ctx.output_folder,
        target_size=target_size,
        total_duration=total_duration,
        clips_to_close=clips_to_close,
        extra_metadata=extra_metadata,
        script_id=ctx.common.get("script_id", ""),
        spec_path=ctx.spec_path,
        started_at=started_at,
    )