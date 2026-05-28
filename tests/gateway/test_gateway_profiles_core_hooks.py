from gateway.config import Platform
from gateway.platforms.base import MessageEvent, should_suppress_gateway_response
from gateway.run import _apply_gateway_runtime_override, _patch_gateway_event_from_hook
from gateway.session import SessionSource


def _event() -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="chat-1",
            chat_type="group",
            user_id="user-1",
        ),
    )


def test_pre_gateway_dispatch_patch_updates_event_metadata_and_context():
    event = _event()

    patched = _patch_gateway_event_from_hook(
        event,
        {
            "action": "allow",
            "text": "rewritten",
            "channel_prompt": "profile prompt",
            "auto_skill": "gateway-profiles:gateway-profiles",
            "metadata": {"reply_suppression": {"sentinel": "NO_REPLY"}},
            "runtime_overrides": {"model": "nous/test"},
        },
    )

    assert patched.text == "rewritten"
    assert patched.channel_prompt == "profile prompt"
    assert patched.auto_skill == "gateway-profiles:gateway-profiles"
    assert patched.metadata["reply_suppression"]["sentinel"] == "NO_REPLY"
    assert patched.metadata["gateway_runtime_overrides"]["model"] == "nous/test"


def test_gateway_runtime_override_applies_model_reasoning_and_toolsets():
    model, runtime, enabled, disabled, reasoning, prompt = _apply_gateway_runtime_override(
        model="old-model",
        runtime_kwargs={"provider": "old", "base_url": None, "api_mode": None},
        enabled_toolsets=["web"],
        disabled_toolsets=None,
        reasoning_config=None,
        combined_ephemeral="base prompt",
        override={
            "model": "new-model",
            "provider": "new-provider",
            "reasoning": "low",
            "toolsets": ["skills", "web"],
            "disabled_toolsets": ["terminal"],
            "channel_prompt": "profile prompt",
        },
    )

    assert model == "new-model"
    assert runtime["provider"] == "new-provider"
    assert enabled == ["skills", "web"]
    assert disabled == ["terminal"]
    assert reasoning == {"enabled": True, "effort": "low"}
    assert prompt == "base prompt\n\nprofile prompt"


def test_gateway_response_suppression_matches_exact_trimmed_sentinel():
    event = _event()

    assert should_suppress_gateway_response(" REPLY_DENIED\n", event)
    assert should_suppress_gateway_response("REPLY_DENIED\nMEDIA:/tmp/out.mp3", event)
    assert not should_suppress_gateway_response("REPLY_DENIED because no trigger", event)
    assert not should_suppress_gateway_response("normal reply", event)


def test_gateway_response_suppression_uses_event_sentinel_and_can_disable():
    event = _event()
    event.metadata["reply_suppression"] = {"sentinel": "NO_REPLY"}

    assert should_suppress_gateway_response("NO_REPLY", event)
    assert not should_suppress_gateway_response("REPLY_DENIED", event)

    event.metadata["reply_suppression"]["enabled"] = False
    assert not should_suppress_gateway_response("NO_REPLY", event)
