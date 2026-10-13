"""Local disk cache for downloaded data plus the provenance record kept beside it.

Every cached artefact is a Parquet file with a sidecar JSON file describing where
the data came from, when it was retrieved and which package versions produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CACHE_DIR


def package_versions() -> dict[str, str]:
    """Record the versions of the packages that materially affect the results."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("yfinance", "pandas", "numpy", "scipy", "statsmodels",
                 "exchange_calendars", "streamlit", "plotly"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover - only hit when an extra is absent
            versions[name] = "not installed"
    return versions


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Provenance:
    """Everything needed to explain and reproduce one cached download."""

    kind: str
    tickers: list[str]
    requested_start: str | None
    requested_end: str | None
    retrieved_at_utc: str
    data_source: str
    adjustment_settings: dict[str, Any]
    package_versions: dict[str, str]
    cache_key: str
    cache_file: str
    file_sha256: str = ""
    actual_start: str | None = None
    actual_end: str | None = None
    row_count: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiskCache:
    """Parquet + JSON sidecar cache rooted at a single directory."""

    def __init__(self, root: Path | str = CACHE_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, kind: str, key: str) -> tuple[Path, Path]:
        folder = self.root / kind
        folder.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)
        return folder / f"{safe}.parquet", folder / f"{safe}.meta.json"

    def exists(self, kind: str, key: str) -> bool:
        data_path, meta_path = self._paths(kind, key)
        return data_path.exists() and meta_path.exists()

    def write(self, kind: str, key: str, frame: pd.DataFrame,
              provenance: Provenance) -> Provenance:
        data_path, meta_path = self._paths(kind, key)
        # Parquet cannot round-trip a tz-aware index with mixed offsets, so the
        # caller is responsible for handing over a storage-safe frame.
        frame.to_parquet(data_path)
        provenance.cache_file = str(data_path)
        provenance.file_sha256 = file_sha256(data_path)
        provenance.row_count = int(len(frame))
        meta_path.write_text(json.dumps(provenance.to_dict(), indent=2))
        return provenance

    def read(self, kind: str, key: str) -> tuple[pd.DataFrame, Provenance] | None:
        data_path, meta_path = self._paths(kind, key)
        if not (data_path.exists() and meta_path.exists()):
            return None
        try:
            frame = pd.read_parquet(data_path)
            meta = json.loads(meta_path.read_text())
        except Exception:
            return None
        return frame, Provenance(**meta)

    def age_seconds(self, kind: str, key: str) -> float | None:
        _, meta_path = self._paths(kind, key)
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            retrieved = datetime.fromisoformat(meta["retrieved_at_utc"])
        except Exception:
            return None
        return (datetime.now(timezone.utc) - retrieved).total_seconds()

    def delete(self, kind: str, key: str) -> None:
        for path in self._paths(kind, key):
            if path.exists():
                path.unlink()


def humanise_age(seconds: float | None) -> str:
    if seconds is None:
        return "no cache"
    if seconds < 90:
        return f"{seconds:.0f} seconds old"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes old"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} hours old"
    return f"{seconds / 86400:.1f} days old"
