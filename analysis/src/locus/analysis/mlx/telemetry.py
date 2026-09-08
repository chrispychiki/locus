"""Full-signal memory telemetry for the server.

Every memory signal the machine exposes, MLX-side and OS-side, uninterpreted: the MLX allocator's self-report (active/peak) and its buffer cache, and the machine-wide picture (wired pages, free, compressor, swap, pressure level). The allocator's self-report alone cannot distinguish allocator pressure from machine-wide distress — wired growth and compression are invisible to it, and they are what a freeze is diagnosed from.

Per-request: MemoryTelemetry samples at 1Hz on a daemon thread and streams JSONL to the memory log (start/sample/end records keyed by request id). end() returns the start/end/max summary the serve log line prints — the stream itself lives on disk, not in payloads.
"""

import json
import re
import subprocess
import threading
import time

import mlx.core as mx

from .paths import MEMORY_LOG_MAX_BYTES, memory_log


def _read(command: list[str]) -> str:
    return subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=5
    ).stdout


def memory_snapshot() -> dict:
    """Reads fail loud: a signal that can't be read is an error, not a zero."""
    out = _read(["vm_stat"])
    page_size = int(re.search(r"page size of (\d+) bytes", out).group(1))
    vm = {}
    for line in out.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            vm[key.strip()] = value.strip().rstrip(".")

    def vm_gib(key):
        return round(int(vm[key]) * page_size / 2**30, 2)

    pressure, swap = (
        line.strip()
        for line in _read(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level", "vm.swapusage"]
        ).splitlines()
    )
    return {
        "mlx_active_gib": round(mx.get_active_memory() / 2**30, 2),
        "mlx_peak_gib": round(mx.get_peak_memory() / 2**30, 2),
        "mlx_cache_gib": round(mx.get_cache_memory() / 2**30, 2),
        "system_wired_gib": vm_gib("Pages wired down"),
        "system_free_gib": vm_gib("Pages free"),
        "system_compressed_gib": vm_gib("Pages occupied by compressor"),
        "pressure_level": int(pressure) if pressure.isdigit() else pressure,
        "swap": swap,
    }


def _roll_if_full() -> None:
    """Roll the log at its size cap, one generation back: a resident server samples every second of every request for as long as it lives, and an unrolled log fills the disk."""
    log = memory_log()
    if not log.exists() or log.stat().st_size < MEMORY_LOG_MAX_BYTES:
        return
    log.replace(log.with_suffix(".jsonl.1"))


class MemoryTelemetry:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self._stop = threading.Event()
        self._max_wired = 0.0
        self.start = self._record("start")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _record(self, kind: str) -> dict:
        snap = memory_snapshot()
        self._max_wired = max(self._max_wired, snap["system_wired_gib"])
        log = memory_log()
        log.parent.mkdir(parents=True, exist_ok=True)
        _roll_if_full()
        with open(log, "a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": int(time.time() * 1000),
                        "request": self.request_id,
                        "kind": kind,
                        **snap,
                    }
                )
                + "\n"
            )
        return snap

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            self._record("sample")

    def end(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=2)
        end = self._record("end")
        return {
            "start": self.start,
            "end": end,
            "max_system_wired_gib": self._max_wired,
        }
