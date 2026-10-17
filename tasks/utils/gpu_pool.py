"""Claim GPUs that nobody is using, shared by every launcher on this machine.

A GPU counts as free when ``nvidia-smi`` lists no compute process on it (from any user), its memory use is below a
small threshold, and no launcher using this module holds a claim on it. A claim is a file created atomically in a
shared directory and records the claiming process id; claims of processes that no longer exist are removed, so a
killed launcher never strands a GPU.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

DEFAULT_CLAIM_DIR = Path(
    os.environ.get("ECHOXFLOW_GPU_CLAIM_DIR") or Path(tempfile.gettempdir()) / "echoxflow-gpu-claims"
)
#: Memory (MiB) above which a GPU with no listed compute process is still treated as busy.
BUSY_MEMORY_MIB = 1024


class GpuPool:
    """Claims free GPUs for one launcher process; ``acquire`` never blocks, the caller decides how to wait."""

    def __init__(self, claim_dir: Path = DEFAULT_CLAIM_DIR, *, busy_memory_mib: int = BUSY_MEMORY_MIB) -> None:
        self.claim_dir = Path(claim_dir)
        self.busy_memory_mib = int(busy_memory_mib)
        self.claimed: set[str] = set()

    def all_gpus(self) -> list[str]:
        return [index for index, _memory in _query_gpus()]

    def acquire(self) -> str | None:
        """Claim one free GPU and return its index, or ``None`` when every GPU is busy or claimed."""
        self.claim_dir.mkdir(parents=True, exist_ok=True)
        busy = _busy_uuids_by_index()
        for index, memory in _query_gpus():
            if index in busy or memory >= self.busy_memory_mib or index in self.claimed:
                continue
            if self._claim(index):
                self.claimed.add(index)
                return index
        return None

    def release(self, index: str) -> None:
        self.claimed.discard(str(index))
        path = self.claim_dir / f"gpu{index}"
        try:
            if path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                path.unlink()
        except OSError:
            pass

    def release_all(self) -> None:
        for index in list(self.claimed):
            self.release(index)

    def _claim(self, index: str) -> bool:
        path = self.claim_dir / f"gpu{index}"
        for _attempt in range(2):
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
            except FileExistsError:
                if not _stale(path):
                    return False
                path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            return True
        return False


def _stale(path: Path) -> bool:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _query_gpus() -> list[tuple[str, int]]:
    rows = _nvidia_smi("--query-gpu=index,memory.used", "--format=csv,noheader,nounits")
    return [(index.strip(), int(float(memory))) for index, memory in rows]


def _busy_uuids_by_index() -> set[str]:
    index_by_uuid = {
        uuid.strip(): index.strip() for index, uuid in _nvidia_smi("--query-gpu=index,uuid", "--format=csv,noheader")
    }
    apps = _nvidia_smi("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader")
    return {index_by_uuid[uuid.strip()] for uuid, _pid in apps if uuid.strip() in index_by_uuid}


def _nvidia_smi(*args: str) -> list[list[str]]:
    output = subprocess.run(["nvidia-smi", *args], check=True, capture_output=True, text=True).stdout
    return [line.split(",") for line in output.splitlines() if line.strip()]
