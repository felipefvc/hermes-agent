---
name: video-analysis
description: "Download social videos or inspect cached video attachments, transcribe audio, sample frames, and summarize with Hermes vision/LLM."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [ffmpeg, ffprobe]
metadata:
  hermes:
    tags: [Video, YouTube, Instagram, Facebook, X, Transcription, Vision, Summarization]
---

# Video Analysis

Use when the user shares a video URL or direct video attachment and asks what it says, shows, claims, summarizes, or implies. URLs are handled through yt-dlp for YouTube, Facebook, Instagram, X/Twitter, and other supported sites. Local attachment paths, such as cached WhatsApp videos, are handled directly.

## Tool

Call `video_download_analyze` with either `url` or `video_path`. The tool downloads or ingests the video, extracts a compact audio track, transcribes it through Hermes STT, samples frames, analyzes frames through Hermes vision, builds a structured understanding of the transcript/frames/metadata/comments, and caches artifacts under `$HERMES_HOME/cache/video_analysis/`.

The tool transcribes and summarizes in the video's source language by default. When relaying the result back to a messaging chat, normally send only `brief_summary` or a short adaptation of it in the language that best fits the chat, usually the language recently used there. Use `detailed_summary`, `key_points`, `visual_evidence`, `transcript_evidence`, and `metadata_comments_context` only when the user asks for details, evidence, fact-checking, or a breakdown.

Default frame sampling is 5 evenly spaced frames:

```json
{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}
```

For a direct video attachment saved by a gateway:

```json
{"video_path": "/root/.hermes/cache/videos/vid_1234.mp4", "source_label": "WhatsApp video attachment"}
```

For denser visual inspection:

```json
{"url": "URL", "frame_mode": "interval", "frame_interval_seconds": 10, "max_frames": 36}
```

For a specific question:

```json
{"url": "URL", "question": "What product is being demonstrated and what claims are made?"}
```

## STT Configuration

The tool reuses Hermes' existing speech-to-text configuration. For an OpenAI-compatible LAN STT service, configure Hermes STT like this:

```yaml
stt:
  enabled: true
  provider: openai
  openai:
    api_key: "local-or-placeholder-key"
    base_url: "http://YOUR-LAN-STT-HOST:PORT/v1"
    model: "whisper-1"
```

Use the model name expected by your local service. If your endpoint ignores API keys, still provide a placeholder key because OpenAI-compatible clients require one.

## Video Tool Configuration

Optional `config.yaml` defaults:

```yaml
plugins:
  enabled:
    - video_analysis

video_analysis:
  include_comments: true
  comments_limit: 25
  # Full-file STT is tried first. If the STT API fails, the tool retries
  # using chunked audio; useful for local Whisper servers with length limits.
  transcription_chunk_seconds: 60
  # auto means do not force a language; let STT detect the video's spoken language.
  transcription_language: auto
  youtube:
    # Optional override. The tool automatically retries YouTube 403/bot
    # download failures with the android player client when unset.
    # player_client: "android"
  cookies_file: "/path/to/cookies.txt"
  # cookies_from_browser: "chrome"
  frames:
    mode: count
    count: 5
    interval_seconds: 10
    max_frames: 24

auxiliary:
  vision:
    provider: auto
  video_analysis:
    provider: auto
    timeout: 120
```

Use `cookies_file` or `cookies_from_browser` for videos that require login. Avoid collecting more comments/frames than needed because each frame analysis consumes vision tokens.

## Cache Behavior

The cache is keyed by source URL for links and by file content for local video attachments. The tool stores:

- original source URL or local source path, canonical URL or local cache id, and extraction date
- downloaded video file
- extracted audio file
- transcript JSON
- sampled frame JPGs
- frame analysis JSON
- final summary JSON
- downloader metadata including description and comments when available

Use `refresh: true` when the video changed, comments should be re-fetched, or you want to recompute summaries with different settings.

## Notes

- Comments are captured only when the site extractor exposes them and authentication permits access.
- `frame_mode: transcript_segments` uses timestamped STT segments when a provider returns them; otherwise it falls back to the configured frame count.
- The normal result omits the full transcript to keep context small. Set `return_transcript: true` only when the user explicitly asks for it.
