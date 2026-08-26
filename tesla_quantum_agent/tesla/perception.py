"""Tesla Perception Layer: resonance detection, LSH, standing-wave cache."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

try:
    import lancedb

    HAS_LANCE = True
except ImportError:
    HAS_LANCE = False

if TYPE_CHECKING:
    from tesla_quantum_agent.core.embedder import QuantizedVibrationalState, VectorEmbedder


@dataclass
class ResonanceResult:
    hit: bool
    content: Optional[str] = None
    vibration: float = 0.0
    energy_cost: float = 0.0
    codes: Optional[np.ndarray] = None
    source: str = "cache"
    similarity: float = 0.0
    qstate: Optional[Any] = None


class RandomProjectionLSH:
    """Locality-sensitive hash via random hyperplanes (not cryptographic hashing)."""

    def __init__(self, dim: int, n_bits: int = 32, seed: int = 369):
        rng = np.random.RandomState(seed)
        planes = rng.randn(n_bits, dim).astype(np.float32)
        planes /= np.linalg.norm(planes, axis=1, keepdims=True) + 1e-8
        self.planes = planes
        self.n_bits = n_bits
        self.buckets: Dict[int, List[str]] = defaultdict(list)

    def hash_vec(self, vec: np.ndarray) -> int:
        bits = (self.planes @ vec.astype(np.float32)) >= 0
        value = 0
        for b in bits:
            value = (value << 1) | int(bool(b))
        return int(value)

    def add(self, key: str, vec: np.ndarray) -> int:
        h = self.hash_vec(vec)
        if key not in self.buckets[h]:
            self.buckets[h].append(key)
        return h

    def neighbors(self, vec: np.ndarray) -> List[str]:
        return list(self.buckets.get(self.hash_vec(vec), []))


class TeslaPerceptionLayer:
    """Dimension 1 – capture at the point of least resistance."""

    def __init__(
        self,
        embedder: "VectorEmbedder",
        resonance_threshold: float = 0.92,
        lsh_bits: int = 32,
        db_path: str = "./tesla_resonance",
    ):
        self.embedder = embedder
        self.resonance_threshold = resonance_threshold
        self.local_models = {
            "resonant": "all-MiniLM-L6-v2",
            "harmonic": "BAAI/bge-small-en-v1.5",
        }
        self.lsh = RandomProjectionLSH(dim=embedder.dim, n_bits=lsh_bits)
        self._memory: Dict[Tuple[int, ...], str] = {}
        self._vectors: Dict[str, np.ndarray] = {}
        self._contents: Dict[str, str] = {}
        self.table = None
        self._db_path = db_path
        if HAS_LANCE:
            try:
                self.db = lancedb.connect(db_path)
                try:
                    self.table = self.db.open_table("standing_waves")
                    if not self._table_has_text_key():
                        print("[perception] standing_waves lacks text_key; dropping for recreate")
                        self.db.drop_table("standing_waves")
                        self.table = None
                        raise RuntimeError("standing_waves missing text_key")
                except Exception as open_exc:
                    print(f"[perception] open_table: {open_exc!r} — creating standing_waves")
                    dim = int(getattr(embedder, "dim", 384) or 384)
                    seed = {
                        "vector": [0.0] * dim,
                        "content": "__seed__",
                        "text_key": "__seed__",
                        "codes": [0],
                        "vibration": 0.0,
                    }
                    self.table = self.db.create_table("standing_waves", data=[seed])
            except Exception as exc:
                print(f"[perception] lancedb connect/create failed: {exc!r}")
                self.table = None
        else:
            print("[perception] lancedb not installed; persist disabled")

    def locality_sensitive_hash(self, text: str) -> str:
        """Stable fingerprint of raw text (exact-match fast path)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _table_has_text_key(self) -> bool:
        if self.table is None:
            return False
        try:
            names = self._table_column_names()
            return "text_key" in names
        except Exception as exc:
            print(f"[perception] schema inspect failed: {exc!r}")
            return False

    def _table_column_names(self) -> List[str]:
        if self.table is None:
            return []
        schema = getattr(self.table, "schema", None)
        names: List[str] = []
        if schema is not None:
            names = list(getattr(schema, "names", []) or [])
            if not names and hasattr(schema, "fields"):
                names = [getattr(f, "name", str(f)) for f in schema.fields]
        if names:
            return names
        arrow = self.table.to_arrow()
        return list(arrow.schema.names)

    def _table_rows(self) -> List[Dict[str, Any]]:
        """Scan standing_waves without pandas."""
        if self.table is None:
            return []
        arrow = self.table.to_arrow()
        rows: List[Dict[str, Any]] = []
        n = arrow.num_rows
        cols = list(arrow.schema.names)
        for i in range(n):
            row: Dict[str, Any] = {}
            for name in cols:
                val = arrow.column(name)[i].as_py()
                row[name] = val
            rows.append(row)
        return rows

    def _unit(self, vec: np.ndarray) -> np.ndarray:
        raw = np.asarray(vec, dtype=np.float32).reshape(-1)
        return raw / (np.linalg.norm(raw) + 1e-9)

    def _lance_lookup_text_key(self, text_key: str, query_vec: np.ndarray) -> Optional[str]:
        """Exact cross-process hit on sha256[:32] of the raw query. Skip __seed__."""
        if self.table is None or text_key == "__seed__":
            return None
        try:
            rows = self._table_rows()
            matches = [
                r for r in rows
                if r.get("text_key") == text_key and r.get("text_key") != "__seed__"
            ]
            print(
                f"[perception] text_key={text_key} arrow_rows={len(rows)} "
                f"matches={len(matches)}"
            )
            if matches:
                content = matches[0].get("content")
                if content is not None and str(content) not in ("", "__seed__"):
                    return str(content)
        except Exception as exc:
            print(f"[perception] arrow text_key lookup failed: {exc!r}")
            try:
                where = f"text_key = '{text_key}'"
                rows = (
                    self.table.search(query_vec.tolist())
                    .where(where, prefilter=True)
                    .limit(4)
                    .to_list()
                )
                print(f"[perception] where text_key lookup rows={len(rows)}")
                for row in rows:
                    if row.get("text_key") == text_key and row.get("text_key") != "__seed__":
                        content = row.get("content")
                        if content is not None and str(content) not in ("", "__seed__"):
                            return str(content)
            except Exception as exc2:
                print(f"[perception] where text_key lookup failed: {exc2!r}")
        return None

    def _dump_miss_diagnostics(self, text_key: str, search_distance: Optional[float] = None) -> None:
        print(f"[perception] miss table_is_none={self.table is None} text_key={text_key}")
        if search_distance is not None:
            print(f"[perception] last ANN _distance={search_distance}")
        if self.table is None:
            return
        try:
            names = self._table_column_names()
            rows = self._table_rows()
            print(f"[perception] columns={names} nrows={len(rows)}")
            keys = [r.get("text_key") for r in rows if r.get("text_key") != "__seed__"]
            print(
                f"[perception] stored_text_keys={keys[:12]} "
                f"exact_match={text_key in set(r.get('text_key') for r in rows)}"
            )
        except Exception as exc:
            print(f"[perception] schema dump failed: {exc!r}")
            try:
                print(f"[perception] table.schema={getattr(self.table, 'schema', None)}")
            except Exception as exc2:
                print(f"[perception] schema attr failed: {exc2!r}")

    def detect_resonance(self, text: str, threshold: Optional[float] = None) -> ResonanceResult:
        threshold = self.resonance_threshold if threshold is None else threshold
        text_key = self.locality_sensitive_hash(text)

        # a) RAM exact-text standing wave: no encode, no VQ, no SAE — true zero energy.
        if text_key in self._contents:
            return ResonanceResult(
                hit=True,
                content=self._contents[text_key],
                vibration=0.0,
                energy_cost=0.0,
                codes=None,
                source="exact-cache",
                similarity=1.0,
                qstate=None,
            )

        # b) Raw MiniLM (or hash) embedding, unit-normalized — stable across processes.
        raw = self._unit(self.embedder.encode(text))

        # c) Persistent exact text_key BEFORE VQ embed (codebooks are process-local).
        if self.table is not None:
            persisted = self._lance_lookup_text_key(text_key, raw)
            if persisted is not None:
                self._contents[text_key] = persisted
                return ResonanceResult(
                    hit=True,
                    content=persisted,
                    vibration=0.0,
                    energy_cost=0.0,
                    codes=None,
                    source="lancedb",
                    similarity=1.0,
                    qstate=None,
                )

        # d) VQ embed only after the persistent exact check misses.
        qstate = self.embedder.embed(text)
        code_key = qstate.code_key()

        # e) RAM discrete-code cache
        if code_key in self._memory:
            return ResonanceResult(
                hit=True,
                content=self._memory[code_key],
                vibration=float(qstate.vibration),
                energy_cost=0.0,
                codes=qstate.codes,
                source="code-cache",
                similarity=1.0,
                qstate=qstate,
            )

        # f) ANN on RAW embedding with cosine (not reconstructed, not default L2).
        search_distance: Optional[float] = None
        if self.table is not None:
            try:
                search = self.table.search(raw.tolist())
                try:
                    search = search.metric("cosine")
                except Exception as metric_exc:
                    print(f"[perception] metric('cosine') failed: {metric_exc!r}")
                    search = None
                if search is not None:
                    results = search.limit(1).to_list()
                    if results:
                        search_distance = float(results[0].get("_distance", 1.0))
                        print(
                            f"[perception] ANN cosine _distance={search_distance} "
                            f"threshold={1 - threshold:.4f} text_key_hit="
                            f"{results[0].get('text_key') == text_key}"
                        )
                        row_key = results[0].get("text_key")
                        content = results[0].get("content")
                        if (
                            search_distance < (1 - threshold)
                            and row_key != "__seed__"
                            and content not in (None, "", "__seed__")
                        ):
                            return ResonanceResult(
                                hit=True,
                                content=str(content),
                                vibration=float(results[0].get("vibration") or qstate.vibration),
                                energy_cost=0.0,
                                codes=qstate.codes,
                                source="lancedb",
                                similarity=1.0 - search_distance,
                                qstate=qstate,
                            )
            except Exception as exc:
                print(f"[perception] lancedb ANN search failed: {exc!r}")

        # g) In-process LSH / brute cosine on raw unit vectors
        candidates = self.lsh.neighbors(raw)
        best_sim, best_content = 0.0, None
        search_keys = candidates or list(self._vectors.keys())
        for key in search_keys:
            stored = self._vectors.get(key)
            if stored is None:
                continue
            sim = float(np.dot(raw, stored) / (
                (np.linalg.norm(raw) * np.linalg.norm(stored)) + 1e-9
            ))
            if sim > best_sim:
                best_sim = sim
                best_content = self._contents.get(key)
        if best_content is not None and best_sim >= threshold:
            return ResonanceResult(
                hit=True,
                content=best_content,
                vibration=float(qstate.vibration),
                energy_cost=0.0,
                codes=qstate.codes,
                source="lsh-cosine",
                similarity=best_sim,
                qstate=qstate,
            )

        # h) miss
        self._dump_miss_diagnostics(text_key, search_distance)
        return ResonanceResult(
            hit=False,
            vibration=float(qstate.vibration),
            energy_cost=float(qstate.original_energy),
            codes=qstate.codes,
            source="miss",
            qstate=qstate,
        )

    def store_standing_wave(
        self,
        text: str,
        content: str,
        qstate: "QuantizedVibrationalState",
    ) -> None:
        code_key = qstate.code_key()
        text_key = self.locality_sensitive_hash(text)
        vec = qstate.reconstructed if qstate.reconstructed is not None else qstate.embedding
        self._memory[code_key] = content
        self._contents[text_key] = content

        raw = qstate.embedding if qstate.embedding is not None else vec
        if raw is not None:
            raw = self._unit(raw)
            self._vectors[text_key] = raw
            self.lsh.add(text_key, raw)

        if self.table is not None and raw is not None:
            try:
                codes = (
                    [int(c) for c in qstate.codes.tolist()]
                    if qstate.codes is not None and len(qstate.codes)
                    else [0]
                )
                self.table.add(
                    [
                        {
                            "vector": raw.tolist(),
                            "content": content,
                            "text_key": text_key,
                            "codes": codes,
                            "vibration": float(qstate.vibration),
                        }
                    ]
                )
                print(f"[perception] lancedb add text_key={text_key} table_is_none=False")
            except Exception as exc:
                print(f"[perception] lancedb add failed: {exc!r}")
        elif self.table is None:
            print("[perception] lancedb add skipped: self.table is None")
