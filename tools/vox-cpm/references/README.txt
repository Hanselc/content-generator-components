Reference voices for VoxCPM Hi-Fi Cloning.

Layout — one subfolder per voice, keyed by referenceId:

  references/
    <referenceId>/
      reference.json          # prompt_audio, prompt_text, reinforce_audio?, reinforce_enabled?
      reference.wav           # prompt clip (paired with prompt_text)
      reference_reinforce.wav # optional reinforce clip (used when reinforce_enabled is true)
    default/                  # fallback used when a request omits referenceId
      ...
    README.txt

reference.json fields:
  - prompt_audio       (required): filename of the prompt clip in this folder
  - prompt_text        (required): exact transcript of the prompt clip
  - reinforce_audio    (optional): filename of the reinforce clip in this folder
  - reinforce_enabled  (optional, default false): when true, reinforce_audio
                        is encoded to latents to reinforce the timbre; when
                        false, the prompt clip is used for both prompt + reference

Guidelines for low-VRAM GPUs (8 GB):
  - Keep reference clips to 5-8 seconds. Longer clips OOM the AudioVAE
    encoder which processes the entire clip in float32 in one pass.
  - Truncate at a silence boundary (word or sentence complete), NOT at an
    arbitrary timestamp. Cutting mid-word corrupts both the audio and the
    transcript match required by Hi-Fi cloning.
  - The prompt_text field MUST be the exact transcript of the (truncated)
    prompt clip. If you shorten the audio, update prompt_text to match.
    Use ASR (e.g. SenseVoice, Whisper) to obtain the transcript if you are
    unsure of the exact words.

To add a new voice:
  1. Create references/<referenceId>/ with the prompt clip (+ optional
     reinforce clip).
  2. Write references/<referenceId>/reference.json with the exact transcript.
  3. No restart is needed for discovery; the server reads the folder on each
     /generate request. Switching the active voice reloads the model.

To re-enable reinforcement for a voice on a GPU with more VRAM:
  1. Truncate the reinforce clip to 5-8s at a silence boundary.
  2. Set "reinforce_enabled": true in that voice's reference.json.
  3. Restart the server (or call /release so the next /generate reloads it).