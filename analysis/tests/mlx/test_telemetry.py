"""Unit tests for the memory telemetry — no model, no server, no GPU work. The snapshot reads the real machine (vm_stat, sysctl): the signals are the point and reading them is free; the JSONL stream writes to a tmp path."""

import json
import subprocess

import pytest
from locus.analysis.mlx import telemetry
from locus.analysis.mlx.telemetry import MemoryTelemetry, memory_snapshot

SIGNALS = {
    "mlx_active_gib",
    "mlx_peak_gib",
    "mlx_cache_gib",
    "system_wired_gib",
    "system_free_gib",
    "system_compressed_gib",
    "pressure_level",
    "swap",
}


class TestMemorySnapshot:
    def test_reads_every_signal_from_the_live_machine(self):
        snap = memory_snapshot()
        assert set(snap) == SIGNALS
        assert snap["system_wired_gib"] > 0
        assert isinstance(snap["pressure_level"], int)
        assert "used" in snap["swap"]

    def test_a_failed_read_is_an_error_not_a_zero(self, monkeypatch):
        """The docstring's contract: a signal that can't be read is an error. A snapshot that swallowed the failure would hand the diagnosis a zero — exactly the wired/compression signals a freeze is read from."""

        def refuse(command):
            raise subprocess.CalledProcessError(1, command)

        monkeypatch.setattr(telemetry, "_read", refuse)
        with pytest.raises(subprocess.CalledProcessError):
            memory_snapshot()


class TestMemoryTelemetry:
    @pytest.fixture(autouse=True)
    def log_in_tmp(self, tmp_path, monkeypatch):
        self.log = tmp_path / "memory.jsonl"
        monkeypatch.setattr(telemetry, "memory_log", lambda: self.log)

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_streams_start_and_end_records_keyed_by_request_id(self):
        t = MemoryTelemetry("req42")
        summary = t.end()
        records = self.records()
        kinds = [r["kind"] for r in records]
        assert kinds[0] == "start" and kinds[-1] == "end"
        assert all(r["request"] == "req42" for r in records)
        assert all(SIGNALS < set(r) for r in records)
        assert summary["max_system_wired_gib"] == max(
            r["system_wired_gib"] for r in records
        )
        assert summary["start"]["system_wired_gib"] == records[0]["system_wired_gib"]
        assert summary["end"]["system_wired_gib"] == records[-1]["system_wired_gib"]

    def test_end_stops_the_sampler(self):
        t = MemoryTelemetry("req43")
        t.end()
        assert not t._thread.is_alive()

    def test_a_full_log_rolls_one_generation_before_the_next_record(self, monkeypatch):
        """A resident server samples every second of every request for as long as it lives; without the roll the log fills the disk."""
        monkeypatch.setattr(telemetry, "MEMORY_LOG_MAX_BYTES", 4096)
        old = "x" * 4096 + "\n"
        self.log.write_text(old)
        MemoryTelemetry("req44").end()
        assert (self.log.parent / "memory.jsonl.1").read_text() == old
        assert all(r["request"] == "req44" for r in self.records())

    def test_under_the_cap_the_log_keeps_accumulating(self):
        MemoryTelemetry("req45").end()
        first = len(self.records())
        MemoryTelemetry("req46").end()
        assert len(self.records()) > first
        assert not (self.log.parent / "memory.jsonl.1").exists()
