"""Repo-bundled gateway profile routing plugin."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_LINK_RE = re.compile(r"https?://\S+", re.IGNORECASE)


@dataclass(frozen=True)
class HarnessDecision:
    action: str
    reason: str = ""
    profile_name: str | None = None
    profile: dict[str, Any] | None = None


class HarnessConfig:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_hermes_home() / "gateway_profiles.yaml"
        self._mtime: float | None = None
        self._data: dict[str, Any] = {}

    def load(self, *, force: bool = False) -> dict[str, Any]:
        if yaml is None:
            logger.warning("gateway_profiles.yaml requires PyYAML")
            return {}
        try:
            stat = self.path.stat()
        except OSError:
            self._mtime = None
            self._data = {}
            return {}
        if not force and self._mtime == stat.st_mtime:
            return self._data
        try:
            loaded = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("Failed to load %s: %s", self.path, exc)
            loaded = {}
        self._mtime = stat.st_mtime
        self._data = loaded if isinstance(loaded, dict) else {}
        return self._data


_CONFIG = HarnessConfig()


def _platform_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


def _source_field(source: Any, name: str) -> str:
    return str(getattr(source, name, "") or "")


def _merged_profile(config: dict[str, Any], profile_name: str) -> dict[str, Any]:
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    profiles = config.get("profiles") if isinstance(config.get("profiles"), dict) else {}
    profile = profiles.get(profile_name) if isinstance(profiles.get(profile_name), dict) else {}
    merged = dict(defaults)
    for key, value in profile.items():
        if key == "triggers" and isinstance(value, dict) and isinstance(merged.get("triggers"), dict):
            trigger_defaults = dict(merged["triggers"])
            trigger_defaults.update(value)
            merged["triggers"] = trigger_defaults
        else:
            merged[key] = value
    return merged


def _binding_matches(binding: dict[str, Any], event: Any) -> bool:
    source = getattr(event, "source", None)
    if source is None:
        return False
    platform = str(binding.get("platform", "") or "").lower()
    if platform and platform not in {"*", _platform_value(getattr(source, "platform", ""))}:
        return False
    chat_id = str(binding.get("chat_id", "") or "")
    if chat_id and chat_id not in {"*", _source_field(source, "chat_id")}:
        return False
    chat_type = str(binding.get("chat_type", "") or "").lower()
    if chat_type and chat_type != _source_field(source, "chat_type").lower():
        return False
    allowed_senders = binding.get("allowed_senders")
    if isinstance(allowed_senders, list) and allowed_senders:
        sender = _source_field(source, "user_id")
        if sender not in {str(item) for item in allowed_senders}:
            return False
    return True


def select_profile(config: dict[str, Any], event: Any) -> tuple[str | None, dict[str, Any] | None]:
    bindings = config.get("bindings")
    if not isinstance(bindings, list):
        return None, None
    for binding in bindings:
        if not isinstance(binding, dict) or not _binding_matches(binding, event):
            continue
        profile_name = str(binding.get("profile") or config.get("default_profile") or "default")
        return profile_name, _merged_profile(config, profile_name)
    return None, None


def _raw_dict(event: Any) -> dict[str, Any]:
    raw = getattr(event, "raw_message", None)
    return raw if isinstance(raw, dict) else {}


def _normalize_messaging_id(value: Any) -> str:
    if not value:
        return ""
    return re.sub(r":\d+(?=@)", "", str(value).strip())


def _reply_to_bot(event: Any) -> bool:
    raw = _raw_dict(event)
    if raw.get("replyToBot") is True:
        return True
    quoted = _normalize_messaging_id(raw.get("quotedParticipant") or raw.get("quotedRemoteJid"))
    bot_ids = {_normalize_messaging_id(item) for item in raw.get("botIds", []) if item}
    return bool(quoted and quoted in bot_ids)


def _mentioned_bot(event: Any) -> bool:
    raw = _raw_dict(event)
    mentioned = {str(item) for item in raw.get("mentionedIds", []) if item}
    bot_ids = {str(item) for item in raw.get("botIds", []) if item}
    return bool(mentioned and bot_ids and mentioned.intersection(bot_ids))


def _alias_mentioned(text: str, aliases: Any) -> bool:
    if not isinstance(aliases, list):
        return False
    for alias in aliases:
        alias_text = str(alias or "").strip()
        if not alias_text:
            continue
        if alias_text.startswith("re:"):
            try:
                if re.search(alias_text[3:], text, re.IGNORECASE):
                    return True
            except re.error:
                continue
        elif re.search(rf"(?<!\w){re.escape(alias_text)}(?!\w)", text, re.IGNORECASE):
            return True
    return False


def should_trigger(profile: dict[str, Any], event: Any) -> tuple[bool, str]:
    source = getattr(event, "source", None)
    chat_type = _source_field(source, "chat_type").lower()
    text = str(getattr(event, "text", "") or "")
    triggers = profile.get("triggers") if isinstance(profile.get("triggers"), dict) else {}

    if chat_type == "dm":
        return bool(triggers.get("dm", True)), "dm"

    if triggers.get("free_response") is True:
        return True, "free_response"
    if bool(triggers.get("reply_to_bot", True)) and _reply_to_bot(event):
        return True, "reply_to_bot"
    if _mentioned_bot(event) or _alias_mentioned(text, triggers.get("mention_aliases")):
        return True, "mention"
    if bool(triggers.get("links", False)) and _LINK_RE.search(text):
        return True, "link"

    group_mode = str(triggers.get("group", "mention_or_reply") or "").lower()
    if group_mode in {"always", "all"}:
        return True, "group_always"
    if group_mode in {"off", "never", "false"}:
        return False, "group_disabled"
    return False, "no_trigger"


def decide_event(config: dict[str, Any], event: Any) -> HarnessDecision:
    if not config or config.get("enabled") is False:
        return HarnessDecision("allow", "disabled")
    profile_name, profile = select_profile(config, event)
    if not profile:
        return HarnessDecision("allow", "unbound")
    triggered, reason = should_trigger(profile, event)
    if not triggered:
        return HarnessDecision("skip", reason, profile_name, profile)
    return HarnessDecision("allow", reason, profile_name, profile)


def _runtime_overrides(profile: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in ("model", "provider", "base_url", "api_mode"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            overrides[key] = value.strip()
    if isinstance(profile.get("reasoning"), (str, dict)):
        overrides["reasoning"] = profile["reasoning"]
    if isinstance(profile.get("toolsets"), list):
        overrides["toolsets"] = profile["toolsets"]
    if isinstance(profile.get("disabled_toolsets"), list):
        overrides["disabled_toolsets"] = profile["disabled_toolsets"]
    return overrides


def _allow_result(decision: HarnessDecision, event: Any) -> dict[str, Any]:
    profile = decision.profile or {}
    prompt_bits = []
    if decision.profile_name:
        prompt_bits.append(f"Gateway profile: {decision.profile_name}")
    prompt = profile.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        prompt_bits.append(prompt.strip())

    gateway_profile_meta = {
        "profile": decision.profile_name,
        "trigger": decision.reason,
    }
    debug_chat_id = profile.get("debug_chat_id") or profile.get("admin_debug_chat_id")
    if isinstance(debug_chat_id, str) and debug_chat_id.strip():
        gateway_profile_meta["debug_chat_id"] = debug_chat_id.strip()
    debug_platform = profile.get("debug_platform") or profile.get("admin_debug_platform")
    if isinstance(debug_platform, str) and debug_platform.strip():
        gateway_profile_meta["debug_platform"] = debug_platform.strip().lower()

    metadata = {
        "gateway_profiles": gateway_profile_meta,
        "reply_suppression": {
            "enabled": profile.get("reply_suppression", True) is not False,
            "sentinel": profile.get("denial_sentinel") or "REPLY_DENIED",
        },
    }
    result: dict[str, Any] = {
        "action": "allow",
        "metadata": metadata,
        "runtime_overrides": _runtime_overrides(profile),
    }
    if prompt_bits:
        result["channel_prompt"] = "\n\n".join(prompt_bits)
    skill = profile.get("skill", profile.get("skills"))
    if isinstance(skill, (str, list)):
        result["auto_skill"] = skill
    return result


def _is_admin(config: dict[str, Any], event: Any) -> bool:
    admins = config.get("admins")
    if not isinstance(admins, list) or not admins:
        return False
    source = getattr(event, "source", None)
    platform = _platform_value(getattr(source, "platform", ""))
    user_id = _source_field(source, "user_id")
    for admin in admins:
        if isinstance(admin, str) and admin == user_id:
            return True
        if not isinstance(admin, dict):
            continue
        admin_platform = str(admin.get("platform", "") or "").lower()
        admin_user = str(admin.get("user_id", "") or "")
        if admin_user == user_id and (not admin_platform or admin_platform == platform):
            return True
    return False


def render_profiles_command(config: dict[str, Any], event: Any, raw_args: str) -> str:
    args = (raw_args or "").strip().split()
    subcommand = args[0].lower() if args else "status"
    if subcommand == "reload":
        _CONFIG.load(force=True)
        return "Gateway profiles config reloaded."
    if subcommand in {"route", "status"}:
        decision = decide_event(config, event)
        source = getattr(event, "source", None)
        lines = [
            "Gateway profiles",
            f"config: {_CONFIG.path}",
            f"chat: {_platform_value(getattr(source, 'platform', ''))}:{_source_field(source, 'chat_id')}",
            f"decision: {decision.action}",
            f"reason: {decision.reason}",
            f"profile: {decision.profile_name or '(none)'}",
        ]
        return "\n".join(lines)
    return "Usage: /profiles [status|route|reload]"


def _maybe_send_command_reply(gateway: Any, event: Any, text: str, config: dict[str, Any] | None = None) -> None:
    adapter_key = getattr(event.source, "platform", None)
    chat_id = event.source.chat_id
    if isinstance(config, dict) and _source_field(event.source, "chat_type").lower() == "group":
        _profile_name, profile = select_profile(config, event)
        if isinstance(profile, dict):
            debug_chat_id = profile.get("debug_chat_id") or profile.get("admin_debug_chat_id")
            if isinstance(debug_chat_id, str) and debug_chat_id.strip():
                chat_id = debug_chat_id.strip()
                debug_platform = profile.get("debug_platform") or profile.get("admin_debug_platform")
                if isinstance(debug_platform, str) and debug_platform.strip():
                    try:
                        adapter_key = type(adapter_key)(debug_platform.strip().lower())
                    except Exception:
                        adapter_key = debug_platform.strip().lower()
    adapter = getattr(gateway, "adapters", {}).get(adapter_key)
    if adapter is None:
        return
    try:
        asyncio.get_running_loop().create_task(adapter.send(chat_id, text))
    except RuntimeError:
        logger.info("Gateway profiles command reply: %s", text)


def pre_gateway_dispatch(event: Any, gateway: Any = None, session_store: Any = None) -> dict[str, Any] | None:
    del session_store
    config = _CONFIG.load()
    text = str(getattr(event, "text", "") or "")
    if text.startswith("/profiles") or text.startswith("/gateway-profiles"):
        if not _is_admin(config, event):
            _maybe_send_command_reply(gateway, event, "Gateway profiles admin access is not configured for this sender.", config)
            return {"action": "skip", "reason": "gateway_profiles_admin_denied"}
        raw_args = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else ""
        _maybe_send_command_reply(gateway, event, render_profiles_command(config, event, raw_args), config)
        return {"action": "skip", "reason": "gateway_profiles_command"}

    decision = decide_event(config, event)
    if decision.action == "skip":
        return {"action": "skip", "reason": f"gateway_profiles:{decision.reason}"}
    if decision.profile:
        return _allow_result(decision, event)
    return None


def gateway_runtime_override(event: Any = None, **kwargs: Any) -> dict[str, Any] | None:
    del kwargs
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    overrides = metadata.get("gateway_runtime_overrides")
    return overrides if isinstance(overrides, dict) else None


def _command_handler(raw_args: str) -> str:
    del raw_args
    return "Use /profiles status, /profiles route, or /profiles reload from a gateway chat."


def register(ctx) -> None:
    plugin_dir = Path(__file__).resolve().parent
    ctx.register_hook("pre_gateway_dispatch", pre_gateway_dispatch)
    ctx.register_hook("gateway_runtime_override", gateway_runtime_override)
    ctx.register_command(
        "profiles",
        _command_handler,
        description="Inspect or reload gateway profile routing",
        args_hint="[status|route|reload]",
    )
    ctx.register_skill(
        "gateway-profiles",
        plugin_dir / "skills" / "gateway-profiles" / "SKILL.md",
        description="Gateway profile denial sentinel and routing behavior.",
    )
