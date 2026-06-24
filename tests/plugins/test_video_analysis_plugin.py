import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _default_summary_path(tools, cache_dir: Path) -> Path:
    config = tools.VideoAnalysisConfig(cache_dir=cache_dir.parent)
    frame_spec = tools._resolve_frame_spec({}, config)
    return cache_dir / f"summary_v{tools.VIDEO_SUMMARY_SCHEMA_VERSION}_{frame_spec['signature']}.json"


def test_register_wires_tool_and_aux_task():
    from plugins.video_analysis import register

    calls = {"tools": [], "aux": [], "hooks": []}

    class Ctx:
        def register_auxiliary_task(self, **kwargs):
            calls["aux"].append(kwargs)

        def register_tool(self, **kwargs):
            calls["tools"].append(kwargs)

        def register_hook(self, *args):
            calls["hooks"].append(args)

    register(Ctx())

    assert calls["aux"][0]["key"] == "video_analysis"
    tool = calls["tools"][0]
    assert tool["name"] == "video_download_analyze"
    assert tool["toolset"] == "video_analysis"
    assert tool["is_async"] is True
    props = tool["schema"]["parameters"]["properties"]
    assert "url" in props
    assert "video_path" in props
    assert "transcription_language" in props
    assert tool["schema"]["parameters"]["required"] == []
    assert [hook[0] for hook in calls["hooks"]] == ["pre_tool_call"]


def test_pre_tool_call_blocks_bare_gateway_video_url():
    from gateway.session_context import clear_session_vars, set_session_vars
    from plugins.video_analysis.tools import maybe_block_implicit_video_tool_call

    tokens = set_session_vars(
        platform="whatsapp",
        current_user_message="https://youtu.be/abc12345678",
    )
    try:
        result = maybe_block_implicit_video_tool_call(
            tool_name="video_download_analyze",
            args={"url": "https://youtu.be/abc12345678"},
        )
    finally:
        clear_session_vars(tokens)

    assert result is not None
    assert result["action"] == "block"


def test_pre_tool_call_allows_explicit_gateway_video_request():
    from gateway.session_context import clear_session_vars, set_session_vars
    from plugins.video_analysis.tools import maybe_block_implicit_video_tool_call

    tokens = set_session_vars(
        platform="whatsapp",
        current_user_message="can you analyze what this video says? https://youtu.be/abc12345678",
    )
    try:
        result = maybe_block_implicit_video_tool_call(
            tool_name="video_download_analyze",
            args={"url": "https://youtu.be/abc12345678"},
        )
    finally:
        clear_session_vars(tokens)

    assert result is None


def test_pre_tool_call_allows_later_reference_without_current_url():
    from gateway.session_context import clear_session_vars, set_session_vars
    from plugins.video_analysis.tools import maybe_block_implicit_video_tool_call

    tokens = set_session_vars(
        platform="whatsapp",
        current_user_message="do that one from before",
    )
    try:
        result = maybe_block_implicit_video_tool_call(
            tool_name="video_download_analyze",
            args={"url": "https://youtu.be/abc12345678"},
        )
    finally:
        clear_session_vars(tokens)

    assert result is None


def test_pre_tool_call_allows_natural_explicit_video_request():
    from gateway.session_context import clear_session_vars, set_session_vars
    from plugins.video_analysis.tools import maybe_block_implicit_video_tool_call

    tokens = set_session_vars(
        platform="whatsapp",
        current_user_message="what's this? https://youtu.be/abc12345678",
    )
    try:
        result = maybe_block_implicit_video_tool_call(
            tool_name="video_download_analyze",
            args={"url": "https://youtu.be/abc12345678"},
        )
    finally:
        clear_session_vars(tokens)

    assert result is None


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


def test_resolve_local_video_path_accepts_agent_visible_cache_path(tmp_path, monkeypatch):
    from plugins.video_analysis import tools

    hermes_home = tmp_path / ".hermes"
    video_dir = hermes_home / "video_cache"
    video_dir.mkdir(parents=True)
    source = video_dir / "vid_quoted.mp4"
    source.write_bytes(b"video")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    resolved = tools._resolve_local_video_path("/root/.hermes/cache/videos/vid_quoted.mp4")

    assert resolved == source


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


def test_frame_spec_signature_busts_old_low_resolution_frame_cache(tmp_path):
    from plugins.video_analysis import tools

    spec = tools._resolve_frame_spec({}, tools.VideoAnalysisConfig(cache_dir=tmp_path))

    assert spec["signature"].startswith("frames-v2_")
    assert spec["extraction_version"] == tools.FRAME_EXTRACTION_SCHEMA_VERSION
    assert spec["analysis_box_size"] == tools.FRAME_ANALYSIS_BOX_SIZE


def test_extract_frame_scales_frames_for_vision(tmp_path, monkeypatch):
    from plugins.video_analysis import tools

    output_path = tmp_path / "frame.jpg"
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        output_path.write_bytes(b"jpeg")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    tools._extract_frame(tmp_path / "source.mp4", output_path, 0.25)

    command = captured["command"]
    scale_filter = (
        f"scale={tools.FRAME_ANALYSIS_BOX_SIZE}:{tools.FRAME_ANALYSIS_BOX_SIZE}:"
        "force_original_aspect_ratio=decrease:flags=lanczos"
    )
    assert "-vf" in command
    assert scale_filter in command


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


def test_youtube_download_403_retries_with_android_player(tmp_path, monkeypatch):
    from plugins.video_analysis import tools

    calls = []
    url = "https://youtu.be/abc12345678"

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
            calls.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url, download):
            assert download is True
            if len(calls) == 1:
                raise RuntimeError("HTTP Error 403: Forbidden")
            (tmp_path / "source.mp4").write_bytes(b"video")
            return {"title": "ok", "webpage_url": url}

    class FakeYtDlp:
        YoutubeDL = FakeYDL

    monkeypatch.setattr(tools, "_ensure_yt_dlp", lambda: FakeYtDlp)

    warnings = []
    result = tools._download_or_reuse_video(
        url,
        tmp_path,
        tools.VideoAnalysisConfig(cache_dir=tmp_path),
        False,
        warnings,
    )

    assert result["success"] is True
    assert len(calls) == 2
    assert calls[0]["extractor_args"]["youtube"].get("player_client") is None
    assert calls[1]["extractor_args"]["youtube"]["player_client"] == ["android"]
    assert calls[1]["extractor_args"]["youtube"]["max_comments"] == ["25"]
    assert "android player client fallback" in warnings[0]


def test_local_video_ingest_caches_metadata_and_manifest(tmp_path, monkeypatch):
    from plugins.video_analysis import tools

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video bytes")
    cache_dir = tmp_path / "analysis-cache"

    monkeypatch.setattr(tools, "_probe_duration", lambda _path: 12.5)

    warnings = []
    result = tools._ingest_or_reuse_local_video(
        source,
        cache_dir,
        False,
        warnings,
        "WhatsApp video attachment",
    )

    cached_video = Path(result["video_path"])
    assert result["success"] is True
    assert cached_video.exists()
    assert cached_video.read_bytes() == b"video bytes"
    assert result["manifest"]["source_type"] == "local_file"
    assert result["manifest"]["source_path"] == str(source.resolve())
    assert result["manifest"]["downloader"] == "local_file"
    assert result["metadata"]["title"] == "clip.mp4"
    assert result["metadata"]["duration"] == 12.5
    assert warnings == []


def test_failed_transcript_cache_retries_with_audio_chunks(tmp_path, monkeypatch):
    from plugins.video_analysis import tools
    from tools import transcription_tools

    video_path = tmp_path / "source.mp4"
    audio_path = tmp_path / "audio.mp3"
    transcript_path = tmp_path / "transcript.json"
    video_path.write_bytes(b"video")
    transcript_path.write_text(
        json.dumps({"success": False, "transcript": "", "error": "old failure"}),
        encoding="utf-8",
    )

    def fake_extract_audio(_video_path, out_path):
        out_path.write_bytes(b"full audio")

    def fake_extract_chunks(_video_path, chunks_dir, refresh, chunk_seconds):
        assert refresh is False
        assert chunk_seconds == 30
        chunks_dir.mkdir(parents=True)
        paths = []
        for idx in range(2):
            path = chunks_dir / f"chunk_{idx:04d}.mp3"
            path.write_bytes(f"chunk {idx}".encode())
            paths.append(path)
        return paths

    calls = []

    def fake_transcribe(path, **kwargs):
        assert kwargs.get("language") == "auto"
        calls.append(path)
        if path.endswith("audio.mp3"):
            return {"success": False, "transcript": "", "error": "API error: Internal Server Error"}
        return {
            "success": True,
            "transcript": f"text from {Path(path).stem}",
            "provider": "openai",
        }

    monkeypatch.setattr(tools, "_extract_audio", fake_extract_audio)
    monkeypatch.setattr(tools, "_extract_audio_chunks", fake_extract_chunks)
    monkeypatch.setattr(tools, "_probe_duration", lambda _path: 30.0)
    monkeypatch.setattr(transcription_tools, "transcribe_audio", fake_transcribe)

    warnings = []
    result = tools._transcribe_or_reuse(
        video_path,
        audio_path,
        transcript_path,
        False,
        warnings,
        30,
    )

    assert result["success"] is True
    assert result["provider"] == "openai"
    assert result["transcript"] == "text from chunk_0000\n\ntext from chunk_0001"
    assert result["segments"][0]["start"] == 0
    assert result["segments"][0]["end"] == 30
    assert result["segments"][1]["start"] == 30
    assert "Ignoring cached failed transcript" in warnings[0]
    assert "Full audio transcription failed" in warnings[1]
    assert calls[0].endswith("audio.mp3")
    assert calls[1].endswith("chunk_0000.mp3")


def test_cached_summary_with_failed_transcription_is_not_reused():
    from plugins.video_analysis.tools import _cached_summary_has_failed_transcription

    assert _cached_summary_has_failed_transcription({
        "success": True,
        "transcription": {
            "success": False,
            "error": "API error: Internal Server Error",
        },
    })
    assert not _cached_summary_has_failed_transcription({
        "success": True,
        "transcription": {"success": True, "error": ""},
    })


@pytest.mark.asyncio
async def test_cached_summary_short_circuits_processing(tmp_path, monkeypatch):
    from plugins.video_analysis import tools

    url = "https://www.youtube.com/watch?v=abc12345678"
    cache_key = tools._cache_key(url)
    cache_dir = tmp_path / cache_key
    cache_dir.mkdir(parents=True)
    summary_path = _default_summary_path(tools, cache_dir)
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
    summary_path = _default_summary_path(tools, cache_dir)
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


@pytest.mark.asyncio
async def test_analyze_accepts_local_video_path(tmp_path, monkeypatch):
    from plugins.video_analysis import tools

    source = tmp_path / "whatsapp_clip.mp4"
    source.write_bytes(b"video bytes")

    monkeypatch.setattr(tools, "check_video_analysis_requirements", lambda: True)
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda _args: tools.VideoAnalysisConfig(cache_dir=tmp_path / "cache"),
    )
    monkeypatch.setattr(tools, "_probe_duration", lambda _path: 8.0)
    monkeypatch.setattr(
        tools,
        "_transcribe_or_reuse",
        lambda *_args: {
            "success": True,
            "transcript": "spoken words",
            "provider": "openai",
        },
    )
    monkeypatch.setattr(
        tools,
        "_extract_or_reuse_frames",
        lambda *_args: {
            "frames_dir": str(tmp_path / "frames"),
            "frames": [{"index": 1, "timestamp_seconds": 4.0, "path": str(tmp_path / "frame.jpg")}],
        },
    )

    async def fake_analyze_frames(*_args):
        return [{"index": 1, "timestamp_seconds": 4.0, "path": str(tmp_path / "frame.jpg"), "analysis": "a scene"}]

    async def fake_summarize(**kwargs):
        assert kwargs["url"] == str(source.resolve())
        return {
            "schema_version": 3,
            "detected_language": "pt",
            "brief_summary": "brief",
            "detailed_summary": "detailed",
            "key_points": ["point"],
            "visual_evidence": ["a scene"],
            "transcript_evidence": ["spoken words"],
            "metadata_comments_context": "local attachment",
            "caveats": [],
        }

    monkeypatch.setattr(tools, "_analyze_or_reuse_frames", fake_analyze_frames)
    monkeypatch.setattr(tools, "_summarize_video", fake_summarize)

    result = await tools.VideoAnalysisService().analyze({
        "video_path": str(source),
        "source_label": "WhatsApp video attachment",
    })

    assert result["success"] is True
    assert result["source_type"] == "local_file"
    assert result["source_url"] == ""
    assert result["source_path"] == str(source.resolve())
    assert result["summary"] == "brief"
    assert result["source_language"] == "pt"
    assert result["brief_summary"] == "brief"
    assert result["detailed_summary"] == "detailed"
    assert result["key_points"] == ["point"]


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
    assert "brief_summary" in prompt
    assert "detailed_summary" in prompt
    assert "detected_language" in prompt
    assert "video's primary spoken/source language" in prompt
    assert "Return only a JSON object" in prompt


def test_video_understanding_normalizes_structured_json():
    from plugins.video_analysis.tools import _normalize_video_understanding

    result = _normalize_video_understanding(
        json.dumps({
            "brief_summary": "short chat answer",
            "detailed_summary": "fuller understanding",
            "detected_language": "es",
            "key_points": ["one", "two"],
            "visual_evidence": ["chart shown"],
            "transcript_evidence": ["speaker makes a claim"],
            "metadata_comments_context": "comments are supportive",
            "caveats": ["sampled frames only"],
        })
    )

    assert result["schema_version"] == 3
    assert result["detected_language"] == "es"
    assert result["brief_summary"] == "short chat answer"
    assert result["detailed_summary"] == "fuller understanding"
    assert result["key_points"] == ["one", "two"]
    assert result["visual_evidence"] == ["chart shown"]
    assert result["transcript_evidence"] == ["speaker makes a claim"]


def test_video_understanding_falls_back_for_plain_text():
    from plugins.video_analysis.tools import _normalize_video_understanding

    result = _normalize_video_understanding("plain summary")

    assert result["brief_summary"] == "plain summary"
    assert result["detailed_summary"] == "plain summary"
    assert "unstructured text" in result["caveats"][0]
