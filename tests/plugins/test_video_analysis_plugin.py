import json

import pytest


def test_register_wires_tool_and_aux_task():
    from plugins.video_analysis import register

    calls = {"tools": [], "aux": []}

    class Ctx:
        def register_auxiliary_task(self, **kwargs):
            calls["aux"].append(kwargs)

        def register_tool(self, **kwargs):
            calls["tools"].append(kwargs)

    register(Ctx())

    assert calls["aux"][0]["key"] == "video_analysis"
    tool = calls["tools"][0]
    assert tool["name"] == "video_analyze"
    assert tool["toolset"] == "video_analysis"
    assert tool["is_async"] is True


def test_check_requirements_needs_ffmpeg_and_ffprobe(monkeypatch):
    from plugins.video_analysis import tools

    monkeypatch.setattr(tools.shutil, "which", lambda name: f"/bin/{name}")
    assert tools.check_video_analysis_requirements() is True

    monkeypatch.setattr(
        tools.shutil,
        "which",
        lambda name: "/bin/ffmpeg" if name == "ffmpeg" else None,
    )
    assert tools.check_video_analysis_requirements() is False


def test_frame_sampling_defaults_evenly_across_duration():
    from plugins.video_analysis.tools import _sample_times

    spec = {
        "mode": "count",
        "count": 5,
        "interval_seconds": 10,
        "max_frames": 24,
    }

    assert _sample_times(60, spec, {}) == [10, 20, 30, 40, 50]


def test_frame_sampling_interval_is_capped():
    from plugins.video_analysis.tools import _sample_times

    spec = {
        "mode": "interval",
        "count": 5,
        "interval_seconds": 10,
        "max_frames": 3,
    }

    assert _sample_times(95, spec, {}) == [0, 10, 20]


def test_frame_sampling_uses_transcript_segments_when_available():
    from plugins.video_analysis.tools import _sample_times

    spec = {
        "mode": "transcript_segments",
        "count": 5,
        "interval_seconds": 10,
        "max_frames": 3,
    }
    transcription = {
        "segments": [
            {"start": 0, "end": 2},
            {"start": 10, "end": 20},
            {"start": 40, "end": 50},
            {"start": 70, "end": 80},
        ]
    }

    assert _sample_times(100, spec, transcription) == [1, 15, 45]


def test_compact_metadata_limits_description_and_comments():
    from plugins.video_analysis.tools import MAX_DESCRIPTION_CHARS, _compact_metadata

    info = {
        "title": "A video",
        "description": "d" * (MAX_DESCRIPTION_CHARS + 50),
        "comments": [
            {"author": "a", "text": "first"},
            {"author": "b", "text": "second"},
            {"author": "c", "text": "third"},
        ],
    }

    metadata = _compact_metadata(info, comments_limit=2)

    assert metadata["title"] == "A video"
    assert len(metadata["description"]) == MAX_DESCRIPTION_CHARS
    assert [c["text"] for c in metadata["comments"]] == ["first", "second"]
    assert metadata["comments_captured"] == 2


@pytest.mark.asyncio
async def test_cached_summary_short_circuits_processing(tmp_path, monkeypatch):
    from plugins.video_analysis import tools

    url = "https://www.youtube.com/watch?v=abc12345678"
    cache_key = tools._cache_key(url)
    cache_dir = tmp_path / cache_key
    cache_dir.mkdir(parents=True)
    summary_path = cache_dir / "summary_count_5_10_24.json"
    summary_path.write_text(
        json.dumps({"success": True, "summary": "cached", "transcript": "full"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(tools, "check_video_analysis_requirements", lambda: True)
    monkeypatch.setattr(tools, "is_safe_url", lambda _url: True)
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda _args: tools.VideoAnalysisConfig(cache_dir=tmp_path),
    )

    result = await tools.VideoAnalysisService().analyze({"url": url})

    assert result["cached"] is True
    assert result["summary"] == "cached"
    assert "transcript" not in result


@pytest.mark.asyncio
async def test_cached_summary_can_return_transcript_from_transcript_cache(tmp_path, monkeypatch):
    from plugins.video_analysis import tools

    url = "https://www.youtube.com/watch?v=abc12345678"
    cache_key = tools._cache_key(url)
    cache_dir = tmp_path / cache_key
    cache_dir.mkdir(parents=True)
    transcript_path = cache_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps({"success": True, "transcript": "full cached transcript"}),
        encoding="utf-8",
    )
    summary_path = cache_dir / "summary_count_5_10_24.json"
    summary_path.write_text(
        json.dumps({
            "success": True,
            "summary": "cached",
            "transcript_path": str(transcript_path),
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(tools, "check_video_analysis_requirements", lambda: True)
    monkeypatch.setattr(tools, "is_safe_url", lambda _url: True)
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda _args: tools.VideoAnalysisConfig(cache_dir=tmp_path),
    )

    result = await tools.VideoAnalysisService().analyze({
        "url": url,
        "return_transcript": True,
    })

    assert result["cached"] is True
    assert result["transcript"] == "full cached transcript"


def test_summary_prompt_includes_surrounding_context():
    from plugins.video_analysis.tools import _summary_prompt

    prompt = _summary_prompt(
        url="https://x.com/user/status/123",
        metadata={
            "title": "Clip title",
            "description": "video description",
            "comments": [{"author": "viewer", "text": "great breakdown"}],
        },
        manifest={"canonical_url": "https://x.com/user/status/123", "extracted_at": "2026-05-30T00:00:00+00:00"},
        transcription={"transcript": "spoken words"},
        frame_analyses=[{"index": 1, "timestamp_seconds": 3, "path": "/tmp/f.jpg", "analysis": "a chart is visible"}],
        question="What is the claim?",
    )

    assert "video description" in prompt
    assert "viewer: great breakdown" in prompt
    assert "spoken words" in prompt
    assert "a chart is visible" in prompt
    assert "What is the claim?" in prompt
