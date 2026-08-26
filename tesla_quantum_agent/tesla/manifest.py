"""Manifestation engine: collapse intelligence into a usable 3D form."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, Optional


class ManifestationEngine:
    """Dimension 8 – content-addressable multi-format output."""

    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _key(self, intelligence: str) -> str:
        return hashlib.md5(intelligence.encode("utf-8")).hexdigest()

    def manifest(self, intelligence: str, target: str = "json") -> Any:
        key = self._key(intelligence)
        cached = key in self.cache
        if cached:
            base = self.cache[key]
        else:
            base = {"core": intelligence, "ts": datetime.now().isoformat(), "key": key}
            self.cache[key] = base

        if target == "markdown":
            return f"# Insight\n\n{base['core']}"
        if target == "api":
            return {"result": base["core"], "cached": cached, "key": key}
        if target == "text":
            return base["core"]
        return dict(base)
