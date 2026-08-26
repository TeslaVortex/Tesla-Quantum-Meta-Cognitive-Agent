"""Field-effect intelligence: one core, many manifestations."""

from __future__ import annotations


class FieldEffectIntelligence:
    """Dimension 5 – one core, many profile modulations (free)."""

    PROFILES = ("executive", "technical", "investor", "poetic", "operator")

    def modulate(self, core: str, profile: str = "technical") -> str:
        text = core or ""
        first = text.split(".")[0].strip() if text else ""
        profiles = {
            "executive": lambda t: (
                (t[:300] + ("…" if len(t) > 300 else ""))
                + "\n\nBottom line: "
                + (first or t[:80])
            ),
            "technical": lambda t: t,
            "investor": lambda t: (
                f"Opportunity: {t[:200]}{'…' if len(t) > 200 else ''}\n"
                "Key metrics embedded: energy, vibration, resonance hit-rate."
            ),
            "poetic": lambda t: (
                "A standing wave collapses into language:\n\n" + t
            ),
            "operator": lambda t: (
                f"[OPERATOR BRIEF]\n{t}\n\n"
                "Act only on residual novelty; cache the rest."
            ),
        }
        fn = profiles.get(profile, profiles["technical"])
        try:
            return fn(text)
        except Exception:
            return text
