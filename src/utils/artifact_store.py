"""
Central artifact store.

Every piece of generated data in this project (poisoned fine-tuning sets,
monitor verdicts, benchmark scores, attack-success-rate results, etc.) should
be written through this module so it all accumulates in one place
(`artifacts/` by default) with a single queryable manifest.

Design goals:
- Append-only: nothing is ever overwritten. New runs just add new files +
  new manifest lines.
- Self-describing: every artifact is tagged with kind/model/generation/
  protocol/meta so you can slice the growing pile of data later without
  remembering file-naming conventions.
- Zero external dependencies beyond the standard library, so it works the
  same whether you're generating data, fine-tuning, monitoring, or evaluating.
"""

from __future__ import annotations

import json
import os
import uuid
import datetime as _dt
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional


def _utcnow_str() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")


def _safe_component(x: Any) -> str:
    """Turn an arbitrary label into a filesystem-safe path component."""
    s = str(x)
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in s)


@dataclass
class ArtifactRecord:
    """One line in artifacts/manifest.jsonl."""

    artifact_id: str
    kind: str
    model: str
    generation: int
    protocol: Optional[str]
    path: str                 # path to the payload file, relative to artifacts/
    created_at: str
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class ArtifactStore:
    """
    Append-only store for everything the pipeline generates.

    Usage:
        store = ArtifactStore()  # or ArtifactStore(root="artifacts")
        store.save(
            kind="poisoned_finetune_data",
            model="qwen3.5-9b",
            generation=0,
            protocol=None,
            data=[...],
            meta={"trigger_year": 2024},
        )

        records = store.query(kind="poisoned_finetune_data", model="qwen3.5-9b")
    """

    MANIFEST_NAME = "manifest.jsonl"

    def __init__(self, root: str | os.PathLike = "artifacts"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / self.MANIFEST_NAME
        if not self.manifest_path.exists():
            self.manifest_path.touch()

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def save(
        self,
        kind: str,
        model: str,
        generation: int,
        data: Any,
        protocol: Optional[str] = None,
        meta: Optional[dict] = None,
        fmt: str = "jsonl",
    ) -> ArtifactRecord:
        """
        Write `data` to artifacts/<kind>/<model>/gen_<generation>[/<protocol>]/
        and append a manifest entry describing it.

        `data` can be:
          - a list of dicts (written as jsonl, one dict per line), or
          - a single dict / any JSON-serializable object (written as json).
        """
        meta = meta or {}
        artifact_id = f"{_utcnow_str()}_{uuid.uuid4().hex[:8]}"

        subdir_parts = [_safe_component(kind), _safe_component(model), f"gen_{generation}"]
        if protocol:
            subdir_parts.append(_safe_component(protocol))
        subdir = self.root.joinpath(*subdir_parts)
        subdir.mkdir(parents=True, exist_ok=True)

        ext = "jsonl" if fmt == "jsonl" else "json"
        payload_path = subdir / f"{artifact_id}.{ext}"

        if fmt == "jsonl":
            if not isinstance(data, Iterable) or isinstance(data, (str, bytes, dict)):
                raise ValueError("fmt='jsonl' expects an iterable of records (e.g. a list of dicts)")
            with payload_path.open("w") as f:
                for row in data:
                    f.write(json.dumps(row) + "\n")
        else:
            with payload_path.open("w") as f:
                json.dump(data, f, indent=2)

        record = ArtifactRecord(
            artifact_id=artifact_id,
            kind=kind,
            model=model,
            generation=generation,
            protocol=protocol,
            path=str(payload_path.relative_to(self.root)),
            created_at=_dt.datetime.utcnow().isoformat() + "Z",
            meta=meta,
        )
        with self.manifest_path.open("a") as f:
            f.write(record.to_json() + "\n")

        return record

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def _iter_manifest(self) -> Iterable[dict]:
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def query(
        self,
        kind: Optional[str] = None,
        model: Optional[str] = None,
        generation: Optional[int] = None,
        protocol: Optional[str] = None,
    ) -> list[dict]:
        """Return manifest entries matching the given filters (None = wildcard)."""
        out = []
        for rec in self._iter_manifest():
            if kind is not None and rec["kind"] != kind:
                continue
            if model is not None and rec["model"] != model:
                continue
            if generation is not None and rec["generation"] != generation:
                continue
            if protocol is not None and rec.get("protocol") != protocol:
                continue
            out.append(rec)
        return out

    def load_payload(self, record: dict) -> Any:
        """Load the actual data referenced by a manifest record."""
        path = self.root / record["path"]
        if path.suffix == ".jsonl":
            rows = []
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        with path.open() as f:
            return json.load(f)

    def load_all(self, **filters) -> list[Any]:
        """Convenience: query + load_payload for every matching record."""
        return [self.load_payload(r) for r in self.query(**filters)]

    def summary(self) -> dict:
        """A quick count of what's in the store, grouped by kind/model/generation."""
        counts: dict[str, int] = {}
        for rec in self._iter_manifest():
            key = f"{rec['kind']} | {rec['model']} | gen={rec['generation']} | protocol={rec.get('protocol')}"
            counts[key] = counts.get(key, 0) + 1
        return counts


if __name__ == "__main__":
    # Tiny smoke test / usage demo.
    store = ArtifactStore(root="artifacts")
    store.save(
        kind="smoke_test",
        model="demo-model",
        generation=0,
        data=[{"hello": "world"}],
        meta={"note": "artifact store self-test"},
    )
    print(json.dumps(store.summary(), indent=2))
