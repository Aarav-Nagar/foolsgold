"""Append-only, hash-chained record of every backtest trial."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

GENESIS = "0" * 64
DIRNAME = ".foolsgold"
ENV_DIR = "FOOLSGOLD_DIR"


def default_dir(start: Optional[str] = None) -> Path:
    """$FOOLSGOLD_DIR, else the nearest ancestor's .foolsgold/, else ./.foolsgold/."""
    env = os.environ.get(ENV_DIR)
    if env:
        return Path(env)
    cur = Path(start or os.getcwd()).resolve()
    for p in [cur] + list(cur.parents):
        if (p / DIRNAME).is_dir():
            return p / DIRNAME
    return cur / DIRNAME


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _digest(prev: str, body: Dict[str, Any]) -> str:
    return hashlib.sha256((prev + "\n" + _canonical(body)).encode("ascii")).hexdigest()


def pack_returns(returns: Sequence[float]) -> bytes:
    a = array("d", returns)
    if sys.byteorder == "big":
        a.byteswap()
    return a.tobytes()


def unpack_returns(data: bytes) -> List[float]:
    a = array("d")
    a.frombytes(data)
    if sys.byteorder == "big":
        a.byteswap()
    return a.tolist()


class _Lock:
    """Cross-process lock so parallel sweeps cannot interleave appends."""

    def __init__(self, path: Path, timeout: float = 30.0, stale: float = 60.0):
        self.path, self.timeout, self.stale = path, timeout, stale

    def __enter__(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                os.close(os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except (FileExistsError, PermissionError):
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        self.path.unlink()
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("could not lock %s" % self.path)
                time.sleep(0.005)

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except OSError:
            pass


class Ledger:
    """One line of JSON per trial. Each line commits to the one before it,
    so deleting or editing a losing trial breaks the chain."""

    def __init__(self, path: Optional[os.PathLike] = None):
        self.dir = Path(path) if path else default_dir()

    @property
    def file(self) -> Path:
        return self.dir / "ledger.jsonl"

    @property
    def returns_dir(self) -> Path:
        return self.dir / "returns"

    def entries(self, study: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.file.exists():
            return []
        out = []
        with open(self.file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    if study is None or e.get("study") == study:
                        out.append(e)
        return out

    def studies(self) -> List[str]:
        seen: Dict[str, None] = {}
        for e in self.entries():
            seen.setdefault(e["study"], None)
        return list(seen)

    def _last(self) -> Optional[Dict[str, Any]]:
        """Last entry, read from the tail so appends stay O(1)."""
        if not self.file.exists():
            return None
        size = self.file.stat().st_size
        if size == 0:
            return None
        with open(self.file, "rb") as fh:
            start = max(0, size - 65536)
            fh.seek(start)
            lines = [ln for ln in fh.read().split(b"\n") if ln.strip()]
        if not lines:
            return None
        if start > 0 and len(lines) < 2:      # one enormous line: fall back to a full read
            entries = self.entries()
            return entries[-1] if entries else None
        return json.loads(lines[-1].decode("utf-8"))

    def head(self) -> str:
        last = self._last()
        return last["hash"] if last else GENESIS

    def _returns_path(self, study: str, returns_sha: str) -> Path:
        tag = hashlib.sha256(study.encode("utf-8")).hexdigest()[:8]
        return self.returns_dir / ("%s-%s.bin" % (tag, returns_sha[:32]))

    def append(self, body: Dict[str, Any], returns: Sequence[float]) -> Dict[str, Any]:
        """Add one trial. `body` holds the caller's fields; chain fields are added here."""
        self.returns_dir.mkdir(parents=True, exist_ok=True)
        blob = pack_returns(returns)
        returns_sha = hashlib.sha256(blob).hexdigest()
        rpath = self._returns_path(body["study"], returns_sha)
        with _Lock(self.dir / "ledger.lock"):
            last = self._last()
            prev = last["hash"] if last else GENESIS
            entry = dict(body)
            entry["seq"] = (last["seq"] + 1) if last else 1
            entry["ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            entry["returns_sha"] = returns_sha
            # Re-running the identical backtest is not a new attempt at finding
            # something, so byte-identical returns in a study count once.
            entry["dup"] = rpath.exists()
            entry["prev"] = prev
            entry["hash"] = _digest(prev, entry)
            if not entry["dup"]:
                rpath.write_bytes(blob)
            with open(self.file, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(_canonical(entry) + "\n")
        return entry

    def load_returns(self, entry: Dict[str, Any]) -> Optional[List[float]]:
        path = self._returns_path(entry["study"], entry["returns_sha"])
        if not path.exists():
            return None
        return unpack_returns(path.read_bytes())

    def verify(self) -> Tuple[bool, List[str]]:
        """Recompute the chain. Returns (ok, problems)."""
        problems: List[str] = []
        prev, seq = GENESIS, 0
        for e in self.entries():
            seq += 1
            body = {k: v for k, v in e.items() if k != "hash"}
            if e.get("seq") != seq:
                problems.append("entry %s: expected seq %d (a trial was removed or reordered)" % (e.get("seq"), seq))
                seq = e.get("seq", seq)
            if e.get("prev") != prev:
                problems.append("entry %s: does not link to the entry before it" % e.get("seq"))
            if _digest(e.get("prev", ""), body) != e.get("hash"):
                problems.append("entry %s: contents were edited after recording" % e.get("seq"))
            prev = e.get("hash", "")
            path = self._returns_path(e["study"], e["returns_sha"])
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != e["returns_sha"]:
                problems.append("entry %s: stored returns do not match the recorded hash" % e.get("seq"))
        return (not problems, problems)
