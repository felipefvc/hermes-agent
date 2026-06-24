from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from plugins.gateway_profiles import decide_event, pre_gateway_dispatch, should_trigger


def _event(
    text: str,
    *,
    chat_type: str = "group",
    raw_message: dict | None = None,
    metadata: dict | None = None,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="group-1",
            chat_type=chat_type,
            user_id="user-1",
            user_name="User",
        ),
        raw_message=raw_message or {},
        metadata=metadata or {},
    )


def _config() -> dict:
    return {
        "enabled": True,
        "defaults": {
            "denial_sentinel": "REPLY_DENIED",
            "skill": "gateway-profiles:gateway-profiles",
            "triggers": {
                "dm": True,
                "group": "mention_or_reply",
                "mention_aliases": ["hermes"],
                "reply_to_bot": True,
                "links": False,
            },
        },
        "profiles": {
            "default": {
                "prompt": "Use the default chat profile.",
                "model": "nous/test-model",
                "reasoning": "low",
                "toolsets": ["web", "skills"],
            }
        },
        "bindings": [
            {
                "platform": "whatsapp",
                "chat_id": "group-1",
                "chat_type": "group",
                "profile": "default",
            }
        ],
    }


def test_group_without_trigger_is_skipped():
    decision = decide_event(_config(), _event("ambient chat"))

    assert decision.action == "skip"
    assert decision.reason == "no_trigger"
    assert decision.profile_name == "default"


def test_alias_mention_triggers_profile():
    decision = decide_event(_config(), _event("hermes please summarize this"))

    assert decision.action == "allow"
    assert decision.reason == "mention"


def test_platform_structural_mention_survives_cleaned_text():
    event = _event(
        "please summarize this",
        metadata={
            "platform_structural_trigger": {
                "platform": "whatsapp",
                "reason": "mention_name",
            }
        },
    )

    assert should_trigger(decision_profile(_config()), event) == (True, "mention_name")


def test_reply_to_bot_triggers_profile():
    event = _event(
        "what do you think?",
        raw_message={
            "quotedParticipant": "bot@s.whatsapp.net",
            "botIds": ["bot@s.whatsapp.net"],
        },
    )

    assert should_trigger(decision_profile(_config()), event) == (True, "reply_to_bot")


def test_reply_to_bot_triggers_from_bridge_flag():
    event = _event(
        "what do you think?",
        raw_message={
            "replyToBot": True,
            "quotedMessageId": "outbound-msg",
        },
    )

    assert should_trigger(decision_profile(_config()), event) == (True, "reply_to_bot")


def test_reply_to_bot_normalizes_whatsapp_device_ids():
    event = _event(
        "what do you think?",
        raw_message={
            "quotedParticipant": "bot:12@s.whatsapp.net",
            "botIds": ["bot@s.whatsapp.net"],
        },
    )

    assert should_trigger(decision_profile(_config()), event) == (True, "reply_to_bot")


def test_link_trigger_is_opt_in():
    profile = decision_profile(_config())

    assert should_trigger(profile, _event("https://example.com")) == (False, "no_trigger")

    profile["triggers"]["links"] = True
    assert should_trigger(profile, _event("https://example.com")) == (True, "link")


def test_pre_dispatch_allow_sets_prompt_skill_runtime_and_suppression(monkeypatch):
    cfg = _config()
    monkeypatch.setattr("plugins.gateway_profiles._CONFIG", SimpleNamespace(load=lambda **_: cfg))

    result = pre_gateway_dispatch(_event("hermes use web"))

    assert result["action"] == "allow"
    assert result["auto_skill"] == "gateway-profiles:gateway-profiles"
    assert "Use the default chat profile." in result["channel_prompt"]
    assert result["runtime_overrides"]["model"] == "nous/test-model"
    assert result["runtime_overrides"]["toolsets"] == ["web", "skills"]
    assert result["metadata"]["reply_suppression"]["sentinel"] == "REPLY_DENIED"


def test_pre_dispatch_sets_cross_platform_debug_target(monkeypatch):
    cfg = _config()
    cfg["profiles"]["default"]["debug_chat_id"] = "telegram-debug-chat"
    cfg["profiles"]["default"]["debug_platform"] = "telegram"
    monkeypatch.setattr("plugins.gateway_profiles._CONFIG", SimpleNamespace(load=lambda **_: cfg))

    result = pre_gateway_dispatch(_event("hermes use web"))

    gateway_profile_meta = result["metadata"]["gateway_profiles"]
    assert gateway_profile_meta["debug_chat_id"] == "telegram-debug-chat"
    assert gateway_profile_meta["debug_platform"] == "telegram"


def decision_profile(config: dict) -> dict:
    decision = decide_event(config, _event("hermes"))
    assert decision.profile is not None
    return decision.profile
