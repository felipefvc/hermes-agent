from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import (
    _is_implicit_video_share_without_request,
    _transcript_has_session_meta,
)
from gateway.session import SessionSource


def _event(text: str = "", **kwargs) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="chat-1",
            chat_type="group",
            user_id="user-1",
        ),
        **kwargs,
    )


def test_bare_youtube_link_is_observed_without_agent_dispatch():
    event = _event("https://youtu.be/siHfHUm3HGE")

    assert _is_implicit_video_share_without_request(event, event.text)


def test_youtube_link_with_explicit_summary_request_dispatches_agent():
    event = _event("Comrad, resuma isso https://youtu.be/siHfHUm3HGE")

    assert not _is_implicit_video_share_without_request(event, event.text)


def test_later_explicit_reference_without_url_dispatches_agent():
    event = _event("Comrad, resuma o vídeo anterior")

    assert not _is_implicit_video_share_without_request(event, event.text)


def test_video_attachment_without_request_is_observed_only():
    event = _event(
        "",
        message_type=MessageType.VIDEO,
        media_urls=["/tmp/clip.mp4"],
        media_types=["video/mp4"],
    )

    assert _is_implicit_video_share_without_request(event, event.text)


def test_video_attachment_with_request_dispatches_agent():
    event = _event(
        "transcreve esse vídeo",
        message_type=MessageType.VIDEO,
        media_urls=["/tmp/clip.mp4"],
        media_types=["video/mp4"],
    )

    assert not _is_implicit_video_share_without_request(event, event.text)


def test_transcript_meta_detection_handles_observed_only_history():
    assert not _transcript_has_session_meta([
        {"role": "user", "content": "https://youtu.be/siHfHUm3HGE", "observed": True},
    ])
    assert _transcript_has_session_meta([
        {"role": "user", "content": "https://youtu.be/siHfHUm3HGE", "observed": True},
        {"role": "session_meta", "tools": []},
    ])
