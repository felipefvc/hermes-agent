"""Agent-facing video analysis tool.

The tool keeps the heavy lifting outside the main agent loop:

* yt-dlp downloads video metadata/media into a stable cache directory.
* ffmpeg extracts a compact audio track and a configurable sample of frames.
* Hermes' existing STT dispatcher transcribes audio, including OpenAI-
  compatible endpoints configured under ``stt.openai``.
* Hermes' existing vision tool analyzes each sampled frame.
* The plugin auxiliary LLM task synthesizes transcript, frame analyses, and
  surrounding metadata into the final video summary.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import unquote, urlparse

from hermes_constants import get_hermes_home
from tools.credential_files import to_host_cache_path
from tools.url_safety import is_safe_url

logger = logging.getLogger(__name__)


class _QuietYtDlpLogger:
    """Keep yt-dlp retries from writing expected fallback noise to stderr."""

    def debug(self, msg: str) -> None:
        logger.debug("yt-dlp: %s", msg)

    def warning(self, msg: str) -> None:
        logger.debug("yt-dlp warning: %s", msg)

    def error(self, msg: str) -> None:
        logger.debug("yt-dlp error: %s", msg)


DEFAULT_FRAME_COUNT = 5
DEFAULT_FRAME_MODE = "count"
DEFAULT_FRAME_INTERVAL_SECONDS = 10.0
DEFAULT_MAX_FRAMES = 24
DEFAULT_COMMENTS_LIMIT = 25
DEFAULT_YOUTUBE_FALLBACK_PLAYER_CLIENT = "android"
DEFAULT_TRANSCRIPTION_CHUNK_SECONDS = 60
DEFAULT_VIDEO_TRANSCRIPTION_LANGUAGE = "auto"
VIDEO_SUMMARY_SCHEMA_VERSION = 3
LOCAL_VIDEO_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".mov",
    ".webm",
    ".mkv",
    ".avi",
    ".mpeg",
    ".mpg",
}
MAX_DESCRIPTION_CHARS = 12000
MAX_COMMENTS_CHARS = 16000
MAX_TRANSCRIPT_PROMPT_CHARS = 30000
MAX_FRAME_ANALYSIS_PROMPT_CHARS = 16000
VIDEO_ANALYSIS_TOOL_NAME = "video_download_analyze"


VIDEO_ANALYZE_SCHEMA: Dict[str, Any] = {
    "name": VIDEO_ANALYSIS_TOOL_NAME,
    "description": (
        "Download and analyze a video URL from YouTube, Facebook, Instagram, "
        "X/Twitter, or another yt-dlp-supported site, or analyze a local "
        "cached video file such as a WhatsApp attachment. Caches the video, "
        "audio transcript, sampled frames, frame vision analyses, metadata, "
        "comments when available, and the final summary under "
        "$HERMES_HOME/cache/video_analysis. Use this when the user shares a "
        "video link or video attachment and asks what it says, shows, claims, "
        "or means."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "The video URL to download and analyze. Provide either "
                    "url or video_path, not both."
                ),
            },
            "video_path": {
                "type": "string",
                "description": (
                    "Absolute local path to a cached video attachment to "
                    "analyze. Provide either video_path or url, not both."
                ),
            },
            "source_label": {
                "type": "string",
                "description": (
                    "Optional human source label for a local video_path, "
                    "for example 'WhatsApp video attachment'."
                ),
            },
            "question": {
                "type": "string",
                "description": (
                    "Optional user question or focus for the analysis. "
                    "Defaults to a general summary."
                ),
            },
            "refresh": {
                "type": "boolean",
                "description": (
                    "Ignore cached summary/frame analysis and recompute. "
                    "Existing media may still be reused unless the download "
                    "metadata needs refreshing. Default false."
                ),
            },
            "frame_mode": {
                "type": "string",
                "enum": ["count", "interval", "transcript_segments"],
                "description": (
                    "Frame sampling strategy. count samples evenly across "
                    "the video; interval captures one frame every "
                    "frame_interval_seconds; transcript_segments uses "
                    "timestamped STT segments when available and otherwise "
                    "falls back to count. Default count."
                ),
            },
            "frame_count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "Number of frames for frame_mode=count. Default 5.",
            },
            "frame_interval_seconds": {
                "type": "number",
                "minimum": 1,
                "description": (
                    "Seconds between sampled frames for frame_mode=interval. "
                    "Default 10."
                ),
            },
            "max_frames": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "description": (
                    "Safety cap for interval or transcript segment sampling. "
                    "Default 24 unless configured otherwise."
                ),
            },
            "include_comments": {
                "type": "boolean",
                "description": (
                    "Ask yt-dlp to collect comments when the extractor "
                    "supports it. Defaults to video_analysis.include_comments "
                    "or true."
                ),
            },
            "comments_limit": {
                "type": "integer",
                "minimum": 0,
                "maximum": 500,
                "description": (
                    "Maximum comments to keep from downloader metadata. "
                    "Default 25."
                ),
            },
            "return_transcript": {
                "type": "boolean",
                "description": (
                    "Include the full transcript in the tool result. Default "
                    "false; transcript_path is always returned."
                ),
            },
            "transcription_language": {
                "type": "string",
                "description": (
                    "Optional STT language override. Default auto omits a "
                    "forced language so the transcript stays in the video's "
                    "spoken language. Use an ISO language code only when you "
                    "need to force one."
                ),
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class VideoAnalysisConfig:
    cache_dir: Path
    frame_mode: str = DEFAULT_FRAME_MODE
    frame_count: int = DEFAULT_FRAME_COUNT
    frame_interval_seconds: float = DEFAULT_FRAME_INTERVAL_SECONDS
    max_frames: int = DEFAULT_MAX_FRAMES
    include_comments: bool = True
    comments_limit: int = DEFAULT_COMMENTS_LIMIT
    cookies_file: str = ""
    cookies_from_browser: str = ""
    youtube_player_client: str = ""
    transcription_chunk_seconds: int = DEFAULT_TRANSCRIPTION_CHUNK_SECONDS
    transcription_language: str = DEFAULT_VIDEO_TRANSCRIPTION_LANGUAGE


def check_video_analysis_requirements() -> bool:
    """Return True when local media processing prerequisites are present."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class VideoAnalysisService:
    """State-light orchestrator for the ``video_download_analyze`` plugin tool."""

    async def handle(self, args: Dict[str, Any], **_kwargs: Any) -> str:
        """Tool handler. Returns a JSON string for the agent loop."""
        try:
            result = await self.analyze(args)
        except Exception as exc:  # noqa: BLE001 - plugin tools must not leak
            logger.warning("%s failed: %s", VIDEO_ANALYSIS_TOOL_NAME, exc, exc_info=True)
            result = {"success": False, "error": str(exc)}
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def analyze(self, args: Dict[str, Any]) -> Dict[str, Any]:
        url = str(args.get("url") or "").strip()
        local_video_arg = str(args.get("video_path") or "").strip()
        if bool(url) == bool(local_video_arg):
            return _error("Provide exactly one of url or video_path")
        if url:
            if not _is_http_url(url):
                return _error("url must be an http(s) video URL")
            if not is_safe_url(url):
                return _error("URL is blocked by Hermes URL safety policy")
            source_path: Optional[Path] = None
            cache_key = _cache_key(url)
        else:
            source_path = _resolve_local_video_path(local_video_arg)
            if source_path is None:
                return _error("video_path must be an existing local video file")
            if source_path.suffix.lower() not in LOCAL_VIDEO_EXTENSIONS:
                return _error(
                    (
                        "video_path has an unsupported extension. Supported "
                        f"extensions: {', '.join(sorted(LOCAL_VIDEO_EXTENSIONS))}"
                    ),
                    video_path=str(source_path),
                )
            cache_key = _file_cache_key(source_path)
        if not check_video_analysis_requirements():
            return _error("video_analysis requires ffmpeg and ffprobe on PATH")

        config = _load_config(args)
        cache_dir = config.cache_dir / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)

        frame_spec = _resolve_frame_spec(args, config)
        question = str(args.get("question") or "").strip()
        summary_signature = f"v{VIDEO_SUMMARY_SCHEMA_VERSION}_{frame_spec['signature']}"
        if question:
            summary_signature = f"{summary_signature}_q{_text_cache_key(question)}"
        summary_path = cache_dir / f"summary_{summary_signature}.json"
        refresh = bool(args.get("refresh", False))
        return_transcript = bool(args.get("return_transcript", False))

        if summary_path.exists() and not refresh:
            cached = _read_json(summary_path, default={})
            if (
                isinstance(cached, dict)
                and cached.get("success")
                and not _cached_summary_has_failed_transcription(cached)
            ):
                cached["cached"] = True
                _shape_transcript_fields(cached, return_transcript)
                return cached

        warnings: List[str] = []

        if url:
            media = await asyncio.to_thread(
                _download_or_reuse_video,
                url,
                cache_dir,
                config,
                refresh,
                warnings,
            )
        else:
            media = await asyncio.to_thread(
                _ingest_or_reuse_local_video,
                source_path,
                cache_dir,
                refresh,
                warnings,
                str(args.get("source_label") or "").strip(),
            )
        if not media.get("success"):
            return media

        video_path = Path(media["video_path"])
        metadata = media.get("metadata") or {}
        manifest = media.get("manifest") or {}
        duration = _coerce_float(metadata.get("duration")) or _probe_duration(video_path)
        source_url = str(manifest.get("source_url") or url or "")
        source_path_text = str(manifest.get("source_path") or "")
        source_reference = source_url or source_path_text or str(video_path)

        audio_path = cache_dir / "audio.mp3"
        transcript_path = cache_dir / "transcript.json"
        transcription = await asyncio.to_thread(
            _transcribe_or_reuse,
            video_path,
            audio_path,
            transcript_path,
            refresh,
            warnings,
            config.transcription_chunk_seconds,
            config.transcription_language,
        )

        frame_manifest = await asyncio.to_thread(
            _extract_or_reuse_frames,
            video_path,
            cache_dir,
            frame_spec,
            duration,
            transcription,
            refresh,
            warnings,
        )

        frame_analysis_path = cache_dir / f"frame_analyses_{frame_spec['signature']}.json"
        frame_analyses = await _analyze_or_reuse_frames(
            frame_manifest,
            frame_analysis_path,
            refresh,
            question,
            warnings,
        )

        video_understanding = await _summarize_video(
            url=source_reference,
            metadata=metadata,
            manifest=manifest,
            transcription=transcription,
            frame_analyses=frame_analyses,
            question=question,
        )
        brief_summary = str(video_understanding.get("brief_summary") or "").strip()
        detailed_summary = str(video_understanding.get("detailed_summary") or "").strip()

        extracted_at = manifest.get("extracted_at") or _utc_now()
        result: Dict[str, Any] = {
            "success": True,
            "cached": False,
            "source_type": manifest.get("source_type") or metadata.get("source_type") or ("url" if url else "local_file"),
            "source_url": source_url,
            "source_path": source_path_text,
            "canonical_url": (
                manifest.get("canonical_url")
                or metadata.get("webpage_url")
                or metadata.get("original_url")
                or source_reference
            ),
            "extracted_at": extracted_at,
            "cache_key": cache_key,
            "cache_dir": str(cache_dir),
            "video_path": str(video_path),
            "audio_path": str(audio_path) if audio_path.exists() else "",
            "metadata_path": str(cache_dir / "metadata.json"),
            "transcript_path": str(transcript_path),
            "frames_dir": str(Path(frame_manifest.get("frames_dir", ""))),
            "frame_analysis_path": str(frame_analysis_path),
            "summary_path": str(summary_path),
            "title": metadata.get("title") or "",
            "uploader": metadata.get("uploader") or metadata.get("channel") or "",
            "duration_seconds": duration,
            "frame_sampling": frame_spec,
            "frames": frame_analyses,
            "transcription": _public_transcription(transcription),
            "summary_schema_version": VIDEO_SUMMARY_SCHEMA_VERSION,
            "source_language": video_understanding.get("detected_language") or "",
            "detected_language": video_understanding.get("detected_language") or "",
            "summary": brief_summary,
            "brief_summary": brief_summary,
            "detailed_summary": detailed_summary,
            "key_points": video_understanding.get("key_points") or [],
            "visual_evidence": video_understanding.get("visual_evidence") or [],
            "transcript_evidence": video_understanding.get("transcript_evidence") or [],
            "metadata_comments_context": video_understanding.get("metadata_comments_context") or "",
            "caveats": video_understanding.get("caveats") or [],
            "video_understanding": video_understanding,
            "surrounding_metadata": _public_metadata(metadata),
            "warnings": warnings,
        }

        transcript_text = str(transcription.get("transcript") or "")
        if return_transcript:
            result["transcript"] = transcript_text
        else:
            result["transcript_excerpt"] = _truncate_middle(transcript_text, 4000)

        _write_json(summary_path, result)
        return result


def _load_config(args: Dict[str, Any]) -> VideoAnalysisConfig:
    cfg: Dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        raw = load_config()
        section = raw.get("video_analysis") if isinstance(raw, dict) else None
        if isinstance(section, dict):
            cfg = section
    except Exception:
        cfg = {}

    frames_cfg = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    youtube_cfg = cfg.get("youtube") if isinstance(cfg.get("youtube"), dict) else {}
    cache_dir_raw = cfg.get("cache_dir") or os.getenv("HERMES_VIDEO_ANALYSIS_CACHE_DIR")
    cache_dir = Path(cache_dir_raw).expanduser() if cache_dir_raw else get_hermes_home() / "cache" / "video_analysis"

    include_comments_default = _coerce_bool(cfg.get("include_comments"), True)
    if "include_comments" in args:
        include_comments_default = bool(args.get("include_comments"))

    return VideoAnalysisConfig(
        cache_dir=cache_dir,
        frame_mode=str(frames_cfg.get("mode") or cfg.get("frame_mode") or DEFAULT_FRAME_MODE),
        frame_count=_coerce_int(frames_cfg.get("count") or cfg.get("frame_count"), DEFAULT_FRAME_COUNT, 1, 100),
        frame_interval_seconds=_coerce_float(
            frames_cfg.get("interval_seconds") or cfg.get("frame_interval_seconds"),
            DEFAULT_FRAME_INTERVAL_SECONDS,
            1.0,
        ),
        max_frames=_coerce_int(frames_cfg.get("max_frames") or cfg.get("max_frames"), DEFAULT_MAX_FRAMES, 1, 200),
        include_comments=include_comments_default,
        comments_limit=_coerce_int(args.get("comments_limit") or cfg.get("comments_limit"), DEFAULT_COMMENTS_LIMIT, 0, 500),
        cookies_file=str(cfg.get("cookies_file") or os.getenv("HERMES_VIDEO_ANALYSIS_COOKIES_FILE") or ""),
        cookies_from_browser=str(cfg.get("cookies_from_browser") or os.getenv("HERMES_VIDEO_ANALYSIS_COOKIES_FROM_BROWSER") or ""),
        youtube_player_client=str(
            youtube_cfg.get("player_client")
            or cfg.get("youtube_player_client")
            or os.getenv("HERMES_VIDEO_ANALYSIS_YOUTUBE_PLAYER_CLIENT")
            or ""
        ).strip(),
        transcription_chunk_seconds=_coerce_int(
            cfg.get("transcription_chunk_seconds"),
            DEFAULT_TRANSCRIPTION_CHUNK_SECONDS,
            10,
            600,
        ),
        transcription_language=str(
            args.get("transcription_language")
            or cfg.get("transcription_language")
            or os.getenv("HERMES_VIDEO_ANALYSIS_TRANSCRIPTION_LANGUAGE")
            or DEFAULT_VIDEO_TRANSCRIPTION_LANGUAGE
        ).strip(),
    )


def _resolve_frame_spec(args: Dict[str, Any], config: VideoAnalysisConfig) -> Dict[str, Any]:
    mode = str(args.get("frame_mode") or config.frame_mode or DEFAULT_FRAME_MODE).strip().lower()
    if mode not in {"count", "interval", "transcript_segments"}:
        mode = DEFAULT_FRAME_MODE
    count = _coerce_int(args.get("frame_count"), config.frame_count, 1, 100)
    interval = _coerce_float(args.get("frame_interval_seconds"), config.frame_interval_seconds, 1.0)
    max_frames = _coerce_int(args.get("max_frames"), config.max_frames, 1, 200)
    signature = _safe_signature(f"{mode}_{count}_{interval:g}_{max_frames}")
    return {
        "mode": mode,
        "count": count,
        "interval_seconds": interval,
        "max_frames": max_frames,
        "signature": signature,
    }


def _resolve_local_video_path(value: str) -> Optional[Path]:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme == "file":
        candidate = to_host_cache_path(unquote(parsed.path))
    else:
        candidate = to_host_cache_path(raw)
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    try:
        if not resolved.is_file():
            return None
    except OSError:
        return None
    return resolved


def _ingest_or_reuse_local_video(
    source_path: Path,
    cache_dir: Path,
    refresh: bool,
    warnings: List[str],
    source_label: str = "",
) -> Dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    metadata_path = cache_dir / "metadata.json"
    existing_manifest = _read_json(manifest_path, default={})
    existing_video_raw = str(existing_manifest.get("video_path") or "")
    existing_video = Path(existing_video_raw) if existing_video_raw else None
    if existing_video is not None and existing_video.exists() and metadata_path.exists() and not refresh:
        _mark_local_source_seen(existing_manifest, manifest_path, source_path, source_label)
        return {
            "success": True,
            "video_path": str(existing_video),
            "metadata": _read_json(metadata_path, default={}),
            "manifest": existing_manifest,
        }

    source_path = source_path.resolve()
    target = cache_dir / f"source{source_path.suffix.lower()}"
    if refresh:
        _delete_source_downloads(cache_dir, keep_path=source_path)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.resolve() == source_path:
            video_path = target
        else:
            shutil.copy2(source_path, target)
            video_path = target
    except Exception as exc:  # noqa: BLE001
        return _error(
            f"Could not cache local video file: {exc}",
            source_path=str(source_path),
            cache_dir=str(cache_dir),
            warnings=warnings,
        )

    metadata = _local_video_metadata(
        source_path=source_path,
        cached_video_path=video_path,
        source_label=source_label,
    )
    _write_json(metadata_path, metadata)

    source_ref = str(source_path)
    manifest = {
        "cache_version": 1,
        "source_type": "local_file",
        "source_url": "",
        "source_path": source_ref,
        "source_label": source_label,
        "canonical_url": f"local-video:{cache_dir.name}",
        "cache_key": cache_dir.name,
        "extracted_at": _utc_now(),
        "video_path": str(video_path),
        "metadata_path": str(metadata_path),
        "downloader": "local_file",
        "seen_sources": [
            {
                "source_path": source_ref,
                "source_label": source_label,
                "seen_at": _utc_now(),
            }
        ],
    }
    _write_json(manifest_path, manifest)
    return {
        "success": True,
        "video_path": str(video_path),
        "metadata": metadata,
        "manifest": manifest,
    }


def _local_video_metadata(
    *,
    source_path: Path,
    cached_video_path: Path,
    source_label: str,
) -> Dict[str, Any]:
    stat = source_path.stat()
    label = source_label or "local video attachment"
    return {
        "id": _file_cache_key(source_path),
        "title": source_path.name,
        "description": (
            f"Local video file from {label}. Original path: {source_path}. "
            f"Cached for analysis at: {cached_video_path}."
        ),
        "uploader": label,
        "webpage_url": "",
        "original_url": "",
        "extractor": "local_file",
        "extractor_key": "LocalFile",
        "duration": _probe_duration(cached_video_path),
        "timestamp": int(stat.st_mtime),
        "comments": [],
        "comments_captured": 0,
        "file_name": source_path.name,
        "source_type": "local_file",
        "source_path": str(source_path),
        "cached_video_path": str(cached_video_path),
        "file_size": stat.st_size,
    }


def _mark_local_source_seen(
    manifest: Dict[str, Any],
    manifest_path: Path,
    source_path: Path,
    source_label: str,
) -> None:
    seen = manifest.get("seen_sources")
    if not isinstance(seen, list):
        seen = []
    source_ref = str(source_path)
    if not any(isinstance(item, dict) and item.get("source_path") == source_ref for item in seen):
        seen.append({
            "source_path": source_ref,
            "source_label": source_label,
            "seen_at": _utc_now(),
        })
    manifest["last_seen_at"] = _utc_now()
    manifest["last_source_path"] = source_ref
    manifest["seen_sources"] = seen[-20:]
    _write_json(manifest_path, manifest)


def _download_or_reuse_video(
    url: str,
    cache_dir: Path,
    config: VideoAnalysisConfig,
    refresh: bool,
    warnings: List[str],
) -> Dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    metadata_path = cache_dir / "metadata.json"
    existing_manifest = _read_json(manifest_path, default={})
    existing_video_raw = str(existing_manifest.get("video_path") or "")
    existing_video = Path(existing_video_raw) if existing_video_raw else None
    if existing_video is not None and existing_video.exists() and metadata_path.exists() and not refresh:
        return {
            "success": True,
            "video_path": str(existing_video),
            "metadata": _read_json(metadata_path, default={}),
            "manifest": existing_manifest,
        }

    yt_dlp = _ensure_yt_dlp()
    if refresh:
        _delete_source_downloads(cache_dir)

    try:
        with yt_dlp.YoutubeDL(_build_ytdlp_opts(cache_dir, config, refresh)) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:  # noqa: BLE001
        if _should_retry_youtube_with_android(url, exc, config):
            warnings.append(
                "Initial YouTube download failed; retried with yt-dlp's "
                "android player client fallback."
            )
            _delete_source_downloads(cache_dir)
            try:
                fallback_opts = _build_ytdlp_opts(
                    cache_dir,
                    config,
                    refresh,
                    youtube_player_client=DEFAULT_YOUTUBE_FALLBACK_PLAYER_CLIENT,
                )
                with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
            except Exception as fallback_exc:  # noqa: BLE001
                return _error(
                    (
                        "yt-dlp could not download this URL after YouTube "
                        f"fallback: {fallback_exc} (initial error: {exc})"
                    ),
                    source_url=url,
                    cache_dir=str(cache_dir),
                    warnings=warnings,
                )
        else:
            return _error(
                f"yt-dlp could not download this URL: {exc}",
                source_url=url,
                cache_dir=str(cache_dir),
                warnings=warnings,
            )

    video_path = _find_downloaded_video(cache_dir)
    if video_path is None:
        return _error(
            "yt-dlp completed but no downloaded video file was found",
            cache_dir=str(cache_dir),
            warnings=warnings,
        )

    metadata = _compact_metadata(info or {}, config.comments_limit)
    if not metadata.get("comments") and config.include_comments:
        warnings.append("No comments were available from the downloader for this URL.")
    _write_json(metadata_path, metadata)

    manifest = {
        "cache_version": 1,
        "source_url": url,
        "canonical_url": metadata.get("webpage_url") or metadata.get("original_url") or url,
        "cache_key": cache_dir.name,
        "extracted_at": _utc_now(),
        "video_path": str(video_path),
        "metadata_path": str(metadata_path),
        "downloader": "yt-dlp",
    }
    _write_json(manifest_path, manifest)
    return {
        "success": True,
        "video_path": str(video_path),
        "metadata": metadata,
        "manifest": manifest,
    }


def _build_ytdlp_opts(
    cache_dir: Path,
    config: VideoAnalysisConfig,
    refresh: bool,
    *,
    youtube_player_client: str = "",
) -> Dict[str, Any]:
    ydl_opts: Dict[str, Any] = {
        "format": "bv*+ba/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": str(cache_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _QuietYtDlpLogger(),
        "continuedl": True,
        "retries": 3,
        "fragment_retries": 3,
        "overwrites": bool(refresh),
    }
    if config.cookies_file:
        ydl_opts["cookiefile"] = config.cookies_file
    if config.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (config.cookies_from_browser, None, None, None)
    if config.include_comments:
        ydl_opts["getcomments"] = True

    extractor_args: Dict[str, Dict[str, List[str]]] = {}
    if config.include_comments and config.comments_limit > 0:
        extractor_args.setdefault("youtube", {})["max_comments"] = [str(config.comments_limit)]

    player_client = (
        youtube_player_client
        or config.youtube_player_client
        or ""
    ).strip()
    if player_client:
        extractor_args.setdefault("youtube", {})["player_client"] = [player_client]

    if extractor_args:
        ydl_opts["extractor_args"] = extractor_args
    return ydl_opts


def _should_retry_youtube_with_android(
    url: str,
    exc: Exception,
    config: VideoAnalysisConfig,
) -> bool:
    if not _is_youtube_url(url):
        return False
    if config.youtube_player_client.strip().lower() == DEFAULT_YOUTUBE_FALLBACK_PLAYER_CLIENT:
        return False
    message = str(exc).lower()
    retry_markers = (
        "http error 403",
        "sign in to confirm",
        "not a bot",
        "sabr",
        "po token",
    )
    return any(marker in message for marker in retry_markers)


def _is_youtube_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return (
        host in {"youtu.be", "youtube.com", "www.youtube.com", "m.youtube.com"}
        or host.endswith(".youtube.com")
    )


def _delete_source_downloads(cache_dir: Path, keep_path: Optional[Path] = None) -> None:
    keep_resolved: Optional[Path] = None
    if keep_path is not None:
        try:
            keep_resolved = keep_path.resolve()
        except OSError:
            keep_resolved = keep_path
    for stale in cache_dir.glob("source.*"):
        if stale.is_file():
            if keep_resolved is not None:
                try:
                    if stale.resolve() == keep_resolved:
                        continue
                except OSError:
                    pass
            stale.unlink(missing_ok=True)


def _transcribe_or_reuse(
    video_path: Path,
    audio_path: Path,
    transcript_path: Path,
    refresh: bool,
    warnings: List[str],
    chunk_seconds: int = DEFAULT_TRANSCRIPTION_CHUNK_SECONDS,
    transcription_language: str = DEFAULT_VIDEO_TRANSCRIPTION_LANGUAGE,
) -> Dict[str, Any]:
    cached = _read_json(transcript_path, default={})
    if cached and cached.get("success") and not refresh:
        return cached
    if cached and not refresh:
        warnings.append("Ignoring cached failed transcript; retrying STT.")

    if not audio_path.exists() or refresh:
        try:
            _extract_audio(video_path, audio_path)
        except RuntimeError as exc:
            warnings.append(str(exc))
            result = {
                "success": False,
                "transcript": "",
                "error": str(exc),
                "provider": "",
                "audio_path": "",
                "transcribed_at": _utc_now(),
            }
            _write_json(transcript_path, result)
            return result

    try:
        from tools.transcription_tools import transcribe_audio

        result = transcribe_audio(str(audio_path), language=transcription_language)
    except Exception as exc:  # noqa: BLE001
        result = {
            "success": False,
            "transcript": "",
            "error": f"STT failed: {exc}",
            "provider": "",
        }
    if not result.get("success"):
        warnings.append(
            "Full audio transcription failed; retrying in smaller audio chunks."
        )
        chunk_result = _transcribe_chunks(
            video_path,
            transcript_path.parent / "audio_chunks",
            refresh,
            max(10, int(chunk_seconds or DEFAULT_TRANSCRIPTION_CHUNK_SECONDS)),
            transcription_language,
        )
        if chunk_result.get("success"):
            result = chunk_result
        else:
            result["chunked_error"] = chunk_result.get("error") or "chunked STT failed"

    result["audio_path"] = str(audio_path)
    result["transcribed_at"] = _utc_now()
    _write_json(transcript_path, result)
    if not result.get("success"):
        warnings.append(str(result.get("error") or "STT returned no transcript"))
    return result


def _transcribe_chunks(
    video_path: Path,
    chunks_dir: Path,
    refresh: bool,
    chunk_seconds: int,
    transcription_language: str = DEFAULT_VIDEO_TRANSCRIPTION_LANGUAGE,
) -> Dict[str, Any]:
    try:
        chunks = _extract_audio_chunks(video_path, chunks_dir, refresh, chunk_seconds)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "transcript": "",
            "error": f"Audio chunk extraction failed: {exc}",
            "provider": "",
        }
    if not chunks:
        return {
            "success": False,
            "transcript": "",
            "error": "No audio chunks were produced for chunked STT fallback",
            "provider": "",
        }

    try:
        from tools.transcription_tools import transcribe_audio
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "transcript": "",
            "error": f"STT unavailable for chunked transcription: {exc}",
            "provider": "",
        }

    transcript_parts: List[str] = []
    segments: List[Dict[str, Any]] = []
    errors: List[str] = []
    provider = ""

    for idx, chunk in enumerate(chunks):
        result = transcribe_audio(str(chunk), language=transcription_language)
        if not provider and result.get("provider"):
            provider = str(result.get("provider"))
        text = str(result.get("transcript") or "").strip()
        if result.get("success") and text:
            start = idx * chunk_seconds
            duration = _probe_duration(chunk) or float(chunk_seconds)
            end = start + duration
            transcript_parts.append(text)
            segments.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "chunk_path": str(chunk),
                "language": result.get("language") or "",
            })
        else:
            errors.append(
                f"{chunk.name}: {result.get('error') or 'empty transcript'}"
            )

    transcript = "\n\n".join(transcript_parts).strip()
    if not transcript:
        return {
            "success": False,
            "transcript": "",
            "error": "Chunked STT fallback produced no transcript: " + "; ".join(errors),
            "provider": provider,
            "segments": [],
        }

    result: Dict[str, Any] = {
        "success": True,
        "transcript": transcript,
        "provider": provider,
        "language": _first_non_empty(
            *(str(segment.get("language") or "") for segment in segments),
        ),
        "segments": segments,
        "chunks_dir": str(chunks_dir),
        "chunk_seconds": chunk_seconds,
        "chunk_count": len(chunks),
    }
    if errors:
        result["warnings"] = errors
    return result


def _extract_or_reuse_frames(
    video_path: Path,
    cache_dir: Path,
    frame_spec: Dict[str, Any],
    duration: Optional[float],
    transcription: Dict[str, Any],
    refresh: bool,
    warnings: List[str],
) -> Dict[str, Any]:
    frames_dir = cache_dir / "frames" / frame_spec["signature"]
    frame_manifest_path = frames_dir / "frames.json"
    cached = _read_json(frame_manifest_path, default={})
    if cached and not refresh:
        frames = cached.get("frames") if isinstance(cached, dict) else None
        if isinstance(frames, list) and all(Path(str(f.get("path", ""))).exists() for f in frames):
            return cached

    if refresh and frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    times = _sample_times(duration, frame_spec, transcription)
    if not times:
        times = [0.0]
    frames: List[Dict[str, Any]] = []
    for idx, timestamp in enumerate(times, start=1):
        out_path = frames_dir / f"frame_{idx:04d}_{_safe_signature(f'{timestamp:.2f}s')}.jpg"
        try:
            _extract_frame(video_path, out_path, timestamp)
            frames.append({
                "index": idx,
                "timestamp_seconds": round(timestamp, 3),
                "path": str(out_path),
            })
        except RuntimeError as exc:
            warnings.append(str(exc))

    manifest = {
        "frames_dir": str(frames_dir),
        "frame_sampling": frame_spec,
        "duration_seconds": duration,
        "frames": frames,
        "created_at": _utc_now(),
    }
    _write_json(frame_manifest_path, manifest)
    return manifest


async def _analyze_or_reuse_frames(
    frame_manifest: Dict[str, Any],
    frame_analysis_path: Path,
    refresh: bool,
    question: Any,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    cached = _read_json(frame_analysis_path, default=None)
    if isinstance(cached, list) and cached and not refresh:
        return cached

    frames = frame_manifest.get("frames") or []
    analyses: List[Dict[str, Any]] = []
    for frame in frames:
        path = str(frame.get("path") or "")
        if not path:
            continue
        prompt = _frame_prompt(frame, len(frames), str(question or ""))
        try:
            from tools.vision_tools import vision_analyze_tool

            raw = await vision_analyze_tool(image_url=path, user_prompt=prompt)
            parsed = _json_loads(raw, default={})
            analysis = parsed.get("analysis") if isinstance(parsed, dict) else str(raw)
            success = bool(parsed.get("success", True)) if isinstance(parsed, dict) else True
        except Exception as exc:  # noqa: BLE001
            analysis = f"Frame vision analysis failed: {exc}"
            success = False
            warnings.append(analysis)
        analyses.append({
            **frame,
            "success": success,
            "analysis": analysis,
        })

    _write_json(frame_analysis_path, analyses)
    return analyses


async def _summarize_video(
    *,
    url: str,
    metadata: Dict[str, Any],
    manifest: Dict[str, Any],
    transcription: Dict[str, Any],
    frame_analyses: List[Dict[str, Any]],
    question: str,
) -> Dict[str, Any]:
    try:
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not load Hermes auxiliary LLM client: {exc}") from exc

    prompt = _summary_prompt(
        url=url,
        metadata=metadata,
        manifest=manifest,
        transcription=transcription,
        frame_analyses=frame_analyses,
        question=question,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You analyze videos from transcript, sampled frames, and social "
                "metadata. Be faithful to the evidence; distinguish what is "
                "seen, what is said, and what surrounding metadata/comments imply. "
                "Return structured JSON so the agent can decide how much detail "
                "to relay to the user."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        response = await async_call_llm(
            task="video_analysis",
            messages=messages,
            temperature=0.2,
            max_tokens=3200,
        )
        text = extract_content_or_reasoning(response)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Video summary LLM failed: {exc}") from exc
    return _normalize_video_understanding(text)


def _normalize_video_understanding(raw: Any) -> Dict[str, Any]:
    raw_text = str(raw or "").strip()
    parsed = raw if isinstance(raw, dict) else _extract_json_object(raw_text)
    if not isinstance(parsed, dict):
        brief = _truncate_middle(raw_text, 1200) if raw_text else "The video analysis completed, but the summarizer returned no text."
        return {
            "schema_version": VIDEO_SUMMARY_SCHEMA_VERSION,
            "detected_language": "",
            "brief_summary": brief,
            "detailed_summary": raw_text or brief,
            "key_points": [],
            "visual_evidence": [],
            "transcript_evidence": [],
            "metadata_comments_context": "",
            "caveats": ["The video summarizer returned unstructured text."],
        }

    detailed = str(parsed.get("detailed_summary") or parsed.get("summary") or "").strip()
    brief = str(parsed.get("brief_summary") or "").strip()
    detected_language = str(
        parsed.get("detected_language")
        or parsed.get("source_language")
        or parsed.get("language")
        or ""
    ).strip()
    if not brief:
        brief = _truncate_middle(detailed, 1200) if detailed else "The video analysis completed, but the summarizer returned no brief summary."
    if not detailed:
        detailed = brief

    return {
        "schema_version": VIDEO_SUMMARY_SCHEMA_VERSION,
        "detected_language": detected_language,
        "brief_summary": brief,
        "detailed_summary": detailed,
        "key_points": _coerce_string_list(parsed.get("key_points")),
        "visual_evidence": _coerce_string_list(parsed.get("visual_evidence")),
        "transcript_evidence": _coerce_string_list(parsed.get("transcript_evidence")),
        "metadata_comments_context": str(parsed.get("metadata_comments_context") or "").strip(),
        "caveats": _coerce_string_list(parsed.get("caveats")),
    }


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    parsed = _json_loads(raw_text, default=None)
    if isinstance(parsed, dict):
        return parsed

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    parsed = _json_loads(raw_text[start : end + 1], default=None)
    return parsed if isinstance(parsed, dict) else None


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    result: List[str] = []
    for item in items:
        if isinstance(item, dict):
            text = json.dumps(item, ensure_ascii=False)
        else:
            text = str(item)
        text = text.strip()
        if text:
            result.append(text)
    return result


def _ensure_yt_dlp():
    try:
        import yt_dlp  # type: ignore

        return yt_dlp
    except ImportError:
        pass
    try:
        from tools.lazy_deps import ensure

        ensure("tool.video_analysis")
        import yt_dlp  # type: ignore

        return yt_dlp
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"yt-dlp is required for video_analysis: {exc}") from exc


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; cannot extract audio")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "32k",
        "-loglevel",
        "error",
        str(audio_path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not audio_path.exists():
        detail = (proc.stderr or proc.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"ffmpeg audio extraction failed: {detail}")


def _extract_audio_chunks(
    video_path: Path,
    chunks_dir: Path,
    refresh: bool,
    chunk_seconds: int,
) -> List[Path]:
    existing = sorted(chunks_dir.glob("chunk_*.mp3"))
    if existing and not refresh:
        return existing

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; cannot extract audio chunks")

    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "32k",
        "-f",
        "segment",
        "-segment_time",
        str(max(10, int(chunk_seconds))),
        "-reset_timestamps",
        "1",
        "-loglevel",
        "error",
        str(chunks_dir / "chunk_%04d.mp3"),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=600)
    chunks = sorted(chunks_dir.glob("chunk_*.mp3"))
    if proc.returncode != 0 or not chunks:
        detail = (proc.stderr or proc.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"ffmpeg audio chunk extraction failed: {detail}")
    return chunks


def _extract_frame(video_path: Path, output_path: Path, timestamp: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; cannot extract frames")
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{max(timestamp, 0.0):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        "-loglevel",
        "error",
        str(output_path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not output_path.exists():
        detail = (proc.stderr or proc.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"ffmpeg frame extraction failed at {timestamp:.2f}s: {detail}")


def _probe_duration(video_path: Path) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return _coerce_float(proc.stdout.strip())


def _sample_times(
    duration: Optional[float],
    frame_spec: Dict[str, Any],
    transcription: Dict[str, Any],
) -> List[float]:
    mode = frame_spec.get("mode")
    max_frames = int(frame_spec.get("max_frames") or DEFAULT_MAX_FRAMES)
    duration_value = _coerce_float(duration)

    if mode == "transcript_segments":
        segment_times = _segment_midpoints(transcription)
        if segment_times:
            return segment_times[:max_frames]

    if mode == "interval":
        interval = _coerce_float(frame_spec.get("interval_seconds"), DEFAULT_FRAME_INTERVAL_SECONDS, 1.0)
        if duration_value and duration_value > 0:
            count = min(max_frames, int(duration_value // interval) + 1)
            return [min(i * interval, max(duration_value - 0.1, 0.0)) for i in range(count)]
        return [0.0]

    count = int(frame_spec.get("count") or DEFAULT_FRAME_COUNT)
    if duration_value and duration_value > 0:
        if count == 1:
            return [max(duration_value / 2.0, 0.0)]
        return [((idx + 1) * duration_value) / (count + 1) for idx in range(count)]
    return [float(i) for i in range(count)]


def _segment_midpoints(transcription: Dict[str, Any]) -> List[float]:
    candidates = transcription.get("segments") or transcription.get("timestamps") or []
    if not isinstance(candidates, list):
        return []
    times: List[float] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        start = _coerce_float(item.get("start"))
        end = _coerce_float(item.get("end"))
        if start is None and end is None:
            continue
        if start is None:
            start = end
        if end is None:
            end = start
        if start is not None and end is not None:
            times.append(max((start + end) / 2.0, 0.0))
    return times


def _compact_metadata(info: Dict[str, Any], comments_limit: int) -> Dict[str, Any]:
    comments = info.get("comments")
    compact_comments: List[Dict[str, Any]] = []
    if isinstance(comments, list) and comments_limit > 0:
        for comment in comments[:comments_limit]:
            if not isinstance(comment, dict):
                continue
            text = str(comment.get("text") or "").strip()
            if not text:
                continue
            compact_comments.append({
                "author": comment.get("author") or comment.get("author_id") or "",
                "text": _truncate_middle(text, 1000),
                "timestamp": comment.get("timestamp"),
                "like_count": comment.get("like_count"),
                "is_favorited": comment.get("is_favorited"),
            })

    keep_keys = [
        "id",
        "title",
        "description",
        "uploader",
        "channel",
        "channel_id",
        "uploader_id",
        "webpage_url",
        "original_url",
        "extractor",
        "extractor_key",
        "duration",
        "upload_date",
        "timestamp",
        "view_count",
        "like_count",
        "repost_count",
        "comment_count",
        "tags",
        "categories",
        "age_limit",
        "availability",
    ]
    metadata = {key: info.get(key) for key in keep_keys if key in info}
    if isinstance(metadata.get("description"), str):
        metadata["description"] = metadata["description"][:MAX_DESCRIPTION_CHARS]
    metadata["comments"] = compact_comments
    metadata["comments_captured"] = len(compact_comments)
    return metadata


def _public_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": metadata.get("title") or "",
        "description_excerpt": _truncate_middle(str(metadata.get("description") or ""), 1000),
        "uploader": metadata.get("uploader") or metadata.get("channel") or "",
        "duration": metadata.get("duration"),
        "upload_date": metadata.get("upload_date"),
        "view_count": metadata.get("view_count"),
        "like_count": metadata.get("like_count"),
        "comment_count": metadata.get("comment_count"),
        "comments_captured": metadata.get("comments_captured", 0),
        "source_type": metadata.get("source_type") or "",
        "source_path": metadata.get("source_path") or "",
    }


def _public_transcription(transcription: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": bool(transcription.get("success")),
        "provider": transcription.get("provider") or "",
        "language": transcription.get("language") or "",
        "error": transcription.get("error") or "",
        "chars": len(str(transcription.get("transcript") or "")),
        "audio_path": transcription.get("audio_path") or "",
        "transcribed_at": transcription.get("transcribed_at") or "",
    }


def _shape_transcript_fields(result: Dict[str, Any], return_transcript: bool) -> None:
    """Apply return_transcript to a cached result in-place."""
    result["return_transcript"] = return_transcript
    transcript_path = Path(str(result.get("transcript_path") or ""))
    if return_transcript:
        if "transcript" not in result and transcript_path.exists():
            cached = _read_json(transcript_path, default={})
            if isinstance(cached, dict):
                result["transcript"] = str(cached.get("transcript") or "")
        return

    result.pop("transcript", None)
    if "transcript_excerpt" not in result and transcript_path.exists():
        cached = _read_json(transcript_path, default={})
        if isinstance(cached, dict):
            result["transcript_excerpt"] = _truncate_middle(
                str(cached.get("transcript") or ""), 4000,
            )


def _cached_summary_has_failed_transcription(cached: Dict[str, Any]) -> bool:
    transcription = cached.get("transcription")
    return (
        isinstance(transcription, dict)
        and transcription.get("success") is False
        and bool(transcription.get("error"))
    )


def _summary_prompt(
    *,
    url: str,
    metadata: Dict[str, Any],
    manifest: Dict[str, Any],
    transcription: Dict[str, Any],
    frame_analyses: List[Dict[str, Any]],
    question: str,
) -> str:
    comments_text = _comments_text(metadata.get("comments") or [])
    frames_text = "\n\n".join(
        (
            f"Frame {frame.get('index')} at {frame.get('timestamp_seconds')}s "
            f"({frame.get('path')}):\n{frame.get('analysis')}"
        )
        for frame in frame_analyses
    )
    transcript = str(transcription.get("transcript") or "")
    stt_error = str(transcription.get("error") or "")
    transcript_language = str(transcription.get("language") or "")
    focus = question or "Give a concise but useful summary of the full video."
    return (
        f"Source URL: {url}\n"
        f"Canonical URL: {manifest.get('canonical_url') or metadata.get('webpage_url') or url}\n"
        f"Extracted at: {manifest.get('extracted_at') or ''}\n"
        f"Title: {metadata.get('title') or ''}\n"
        f"Uploader/channel: {metadata.get('uploader') or metadata.get('channel') or ''}\n"
        f"Duration seconds: {metadata.get('duration') or ''}\n"
        f"Upload date: {metadata.get('upload_date') or ''}\n"
        f"Stats: views={metadata.get('view_count') or ''}, likes={metadata.get('like_count') or ''}, comments={metadata.get('comment_count') or ''}\n\n"
        f"Transcript language reported by STT: {transcript_language or '(auto/unknown)'}\n\n"
        f"User focus/question:\n{focus}\n\n"
        f"Video description / surrounding text:\n"
        f"{_truncate_middle(str(metadata.get('description') or ''), MAX_DESCRIPTION_CHARS)}\n\n"
        f"Sampled user comments captured by the downloader:\n"
        f"{_truncate_middle(comments_text, MAX_COMMENTS_CHARS) or '(none available)'}\n\n"
        f"Audio transcript"
        f"{' (STT error: ' + stt_error + ')' if stt_error and not transcript else ''}:\n"
        f"{_truncate_middle(transcript, MAX_TRANSCRIPT_PROMPT_CHARS) or '(no transcript available)'}\n\n"
        f"Sampled frame analyses:\n"
        f"{_truncate_middle(frames_text, MAX_FRAME_ANALYSIS_PROMPT_CHARS) or '(no frame analyses available)'}\n\n"
        "Return only a JSON object with these keys:\n"
        "- brief_summary: 1-2 short paragraphs suitable for a messaging chat.\n"
        "- detailed_summary: a fuller evidence-aware understanding of the video.\n"
        "- detected_language: BCP-47/ISO language code or concise language name for the video's primary spoken/source language.\n"
        "- key_points: array of concise strings.\n"
        "- visual_evidence: array of concise strings about what sampled frames show.\n"
        "- transcript_evidence: array of concise strings about important spoken/audio claims.\n"
        "- metadata_comments_context: concise string summarizing description/comments/context.\n"
        "- caveats: array of concise strings for uncertainty, missing transcript, sparse frames, or unavailable comments.\n\n"
        "Write brief_summary, detailed_summary, and evidence/context fields in "
        "the video's primary spoken/source language by default, preserving the "
        "language of the transcript rather than translating to English. If the "
        "transcript is unavailable, infer the source language from visible text, "
        "metadata, comments, or the user's question and note uncertainty in caveats. "
        "If the user asked a specific question, answer it first in brief_summary "
        "and address it thoroughly in detailed_summary. Do not wrap the JSON in "
        "Markdown fences."
    )


def _frame_prompt(frame: Dict[str, Any], total_frames: int, question: str) -> str:
    focus = f"\nUser focus: {question.strip()}" if question.strip() else ""
    return (
        f"This is sampled frame {frame.get('index')} of {total_frames} from a video, "
        f"at approximately {frame.get('timestamp_seconds')} seconds.{focus}\n\n"
        "Describe the visible scene, people/objects, actions, on-screen text, "
        "setting, visual style, and anything that would help summarize the video. "
        "Be concise but specific."
    )


def _comments_text(comments: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, comment in enumerate(comments, start=1):
        author = comment.get("author") or "unknown"
        text = str(comment.get("text") or "").strip()
        if text:
            lines.append(f"{idx}. {author}: {text}")
    return "\n".join(lines)


def _find_downloaded_video(cache_dir: Path) -> Optional[Path]:
    ignored_suffixes = {".json", ".part", ".ytdl", ".temp", ".tmp"}
    candidates = [
        path
        for path in cache_dir.glob("source.*")
        if path.is_file() and path.suffix.lower() not in ignored_suffixes
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:32]


def _file_cache_key(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"local-video-v1\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:32]


def _text_cache_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:12]


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _safe_signature(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "default"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _coerce_int(value: Any, default: int = 0, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(result, minimum)
    if maximum is not None:
        result = min(result, maximum)
    return result


def _coerce_float(value: Any, default: Optional[float] = None, minimum: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        result = max(result, minimum)
    return result


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _first_non_empty(*values: str) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars]
    head = max_chars // 2
    tail = max_chars - head - 20
    return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_loads(raw: Any, default: Any = None) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except Exception:
        return default


def _error(message: str, **extra: Any) -> Dict[str, Any]:
    return {"success": False, "error": message, **extra}
