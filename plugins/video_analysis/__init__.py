"""Video analysis plugin.

Adds one agent-facing tool, ``video_download_analyze``, for downloading and analyzing
videos from YouTube, Facebook, Instagram, X/Twitter, other yt-dlp-backed sites,
and local cached video attachments.
"""

from __future__ import annotations

from plugins.video_analysis.tools import (
    VIDEO_ANALYZE_SCHEMA,
    VIDEO_ANALYSIS_TOOL_NAME,
    VideoAnalysisService,
    check_video_analysis_requirements,
)


def register(ctx) -> None:
    """Register the video analysis tool and its auxiliary summarizer task."""
    ctx.register_auxiliary_task(
        key="video_analysis",
        display_name="Video analysis",
        description="social video transcript/frame synthesis",
        defaults={
            "provider": "auto",
            "model": "",
            "timeout": 120,
            "temperature": 0.2,
            "extra_body": {},
        },
    )
    service = VideoAnalysisService()
    ctx.register_tool(
        name=VIDEO_ANALYSIS_TOOL_NAME,
        toolset="video_analysis",
        schema=VIDEO_ANALYZE_SCHEMA,
        handler=service.handle,
        check_fn=check_video_analysis_requirements,
        is_async=True,
        description="When explicitly requested, download, transcribe, sample frames, and summarize a social video URL or local video file.",
    )
