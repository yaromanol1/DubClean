"""Explicit policy for settings used by the one-click full-film workflow."""
from __future__ import annotations

from typing import Any


VALID_POLICIES = {"auto", "on", "off"}


def normalize_policy(value: Any, *, setting_name: str) -> str:
    """Return a stable policy value or reject an ambiguous API request."""
    policy = str(value or "auto").strip().lower()
    aliases = {
        "automatic": "auto",
        "always": "on",
        "enabled": "on",
        "never": "off",
        "disabled": "off",
    }
    policy = aliases.get(policy, policy)
    if policy not in VALID_POLICIES:
        raise ValueError(
            f"Неизвестный режим «{setting_name}». "
            "Выберите: автоматически, включить или выключить."
        )
    return policy


def resolve_policies(
    parameters: dict[str, Any],
    recommendations: dict[str, Any],
) -> dict[str, Any]:
    """Resolve user policy and analysis output into actual pipeline settings.

    ``auto`` follows the current compatibility analysis. ``on`` and ``off``
    are explicit user overrides and therefore never inherit the recommendation.
    The algorithmic reference adapter still retains its own low-confidence
    safety gate: requesting it cannot manufacture an unsafe profile.
    """
    alignment_policy = normalize_policy(
        parameters.get("reference_alignment_policy"),
        setting_name="выравнивание качества",
    )
    second_pass_policy = normalize_policy(
        parameters.get("speech_second_pass_policy"),
        setting_name="повторный проход MossFormer",
    )

    recommended_alignment = str(
        recommendations.get("recommended_alignment") or "none"
    ).strip().lower()
    alignment_recommended = recommended_alignment in {
        "algorithmic",
        "local_auto",
    }
    alignment_enabled = (
        alignment_policy == "on"
        or (alignment_policy == "auto" and alignment_recommended)
    )

    recommended_second_pass = bool(
        recommendations.get("recommended_speech_extraction_second_pass", False)
    )
    second_pass_enabled = (
        second_pass_policy == "on"
        or (second_pass_policy == "auto" and recommended_second_pass)
    )

    return {
        "reference_alignment_policy": alignment_policy,
        "speech_second_pass_policy": second_pass_policy,
        "recommended_alignment": recommended_alignment,
        "recommended_speech_extraction_second_pass": recommended_second_pass,
        "reference_adapter": "algorithmic" if alignment_enabled else "raw",
        "speech_extraction_second_pass": second_pass_enabled,
    }
