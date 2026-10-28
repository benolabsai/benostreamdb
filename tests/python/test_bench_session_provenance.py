"""Tests for the benchmark session-provenance helpers.

The DMI fixture is real output from the benchmark host (`sudo dmidecode -t
memory`), which is the only way to be sure the parser handles what `dmidecode`
actually prints: 4× 32 GB G.Skill F4-3600C18-32GVK, DDR4-3600, 2R, 1.2 V.

The fixture also demonstrates the limitation the calibration probes exist for:
DMI has no tCL/tRCD/tRP field, so a manual timing change is invisible here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
MODULE_PATH = BENCHMARKS_DIR / "session_provenance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("session_provenance", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["session_provenance"] = module
    spec.loader.exec_module(module)
    return module


sp = _load_module()


DMIDECODE_MEMORY = """# dmidecode 3.6
Getting SMBIOS data from sysfs.
SMBIOS 3.3.0 present.

Handle 0x003B, DMI type 16, 23 bytes
Physical Memory Array
\tLocation: System Board Or Motherboard
\tUse: System Memory
\tError Correction Type: None
\tMaximum Capacity: 128 GB
\tError Information Handle: 0x003A
\tNumber Of Devices: 4

Handle 0x0043, DMI type 17, 84 bytes
Memory Device
\tArray Handle: 0x003B
\tError Information Handle: 0x0042
\tTotal Width: 64 bits
\tData Width: 64 bits
\tSize: 32 GB
\tForm Factor: DIMM
\tSet: None
\tLocator: DIMM_A1
\tBank Locator: BANK 0
\tType: DDR4
\tType Detail: Synchronous Unbuffered (Unregistered)
\tSpeed: 3600 MT/s
\tManufacturer: G Skill Intl
\tSerial Number: 00000000
\tAsset Tag: Not Specified
\tPart Number: F4-3600C18-32GVK
\tRank: 2
\tConfigured Memory Speed: 3600 MT/s
\tMinimum Voltage: 1.2 V
\tMaximum Voltage: 1.2 V
\tConfigured Voltage: 1.2 V
\tMemory Technology: DRAM
\tMemory Operating Mode Capability: Volatile memory
\tFirmware Version: Unknown
\tModule Manufacturer ID: Bank 5, Hex 0xCD
\tModule Product ID: Unknown
\tMemory Subsystem Controller Manufacturer ID: Unknown
\tNon-Volatile Size: None
\tVolatile Size: 32 GB
\tCache Size: None
\tLogical Size: None

Handle 0x0046, DMI type 17, 84 bytes
Memory Device
\tSize: 32 GB
\tLocator: DIMM_A2
\tType: DDR4
\tSpeed: 3600 MT/s
\tPart Number: F4-3600C18-32GVK
\tRank: 2
\tConfigured Memory Speed: 3600 MT/s
\tConfigured Voltage: 1.2 V

Handle 0x0049, DMI type 17, 84 bytes
Memory Device
\tSize: 32 GB
\tLocator: DIMM_B1
\tType: DDR4
\tSpeed: 3600 MT/s
\tPart Number: F4-3600C18-32GVK
\tRank: 2
\tConfigured Memory Speed: 3600 MT/s
\tConfigured Voltage: 1.2 V

Handle 0x004C, DMI type 17, 84 bytes
Memory Device
\tSize: 32 GB
\tLocator: DIMM_B2
\tType: DDR4
\tSpeed: 3600 MT/s
\tPart Number: F4-3600C18-32GVK
\tRank: 2
\tConfigured Memory Speed: 3600 MT/s
\tConfigured Voltage: 1.2 V
"""


def test_parse_dmidecode_memory_summarizes_the_installed_modules():
    summary = sp.parse_dmidecode_memory(DMIDECODE_MEMORY)

    assert summary is not None
    assert summary.startswith("4× 32 GB DDR4 @ 3600 MT/s")
    assert "2R" in summary
    assert "1.2 V" in summary
    assert "F4-3600C18-32GVK" in summary
    # Every populated slot is named, so a moved or failed module is visible.
    for locator in ("DIMM_A1", "DIMM_A2", "DIMM_B1", "DIMM_B2"):
        assert locator in summary


def test_parse_dmidecode_memory_ignores_empty_slots():
    empty = "Memory Device\n\tSize: No Module Installed\n\tLocator: DIMM_A1\n"
    assert sp.parse_dmidecode_memory(empty) is None


def test_parse_dmidecode_memory_reports_configured_speed_not_rated_speed():
    """A downclocked kit must not look identical to one running at its rating."""
    text = (
        "Memory Device\n"
        "\tSize: 32 GB\n"
        "\tType: DDR4\n"
        "\tSpeed: 3600 MT/s\n"
        "\tConfigured Memory Speed: 2666 MT/s\n"
    )
    summary = sp.parse_dmidecode_memory(text)
    assert summary is not None
    assert "2666 MT/s" in summary
    assert "3600 MT/s" not in summary


def test_parse_decode_dimms_timings_accepts_packed_and_labelled_forms():
    packed = "  Timings: 18-22-22-42 (tCL-tRCD-tRP-tRAS)\n"
    assert sp.parse_decode_dimms_timings(packed) == "18-22-22"

    labelled = "  tCL: 18\n  tRCD: 22\n  tRP: 22\n"
    assert sp.parse_decode_dimms_timings(labelled) == "18-22-22"


def test_parse_decode_dimms_timings_is_none_without_data():
    assert sp.parse_decode_dimms_timings("no timings here") is None


def test_fingerprint_id_ignores_hostname_and_timestamp():
    base = {
        "cpu_model": "AMD Ryzen 9 5900XT 16-Core Processor",
        "cpu_cores": 32,
        "mem_total_bytes": 130_000_000_000,
        "kernel": "7.0.0-38-generic",
        "governor": "powersave",
        "cpu_freq_limits": {"driver": "amd-pstate-epp"},
        "dimm_config": "4× 32 GB DDR4 @ 3600 MT/s",
        "dimm_timings": "18-22-22",
        "engine_version": "0.14.0",
    }
    same = dict(base, hostname="other-box", platform="Linux-other", python="3.13.0")
    assert sp.fingerprint_id(base) == sp.fingerprint_id(same)


def test_fingerprint_id_changes_with_memory_timings():
    base = {
        "cpu_model": "AMD Ryzen 9 5900XT 16-Core Processor",
        "dimm_timings": "18-22-22",
    }
    tightened = dict(base, dimm_timings="16-19-19")
    assert sp.fingerprint_id(base) != sp.fingerprint_id(tightened)


def test_verify_detects_sections_from_a_different_session():
    ledger = {
        "schema": 1,
        "sessions": {
            "old": {"fingerprint": {}, "first_seen": "2026-10-11T00:00:00Z"},
            "new": {"fingerprint": {}, "first_seen": "2026-10-28T00:00:00Z"},
        },
        "sections": {
            "vector": {"session": "old"},
            "graph": {"session": "new"},
            "sql": {"session": "old"},
        },
    }
    primary, mismatches = sp.verify(ledger)
    assert primary == "old"
    assert mismatches == [("graph", "new")]


def test_verify_accepts_a_single_session_and_unstamped_legacy_artifacts():
    ledger = {
        "schema": 1,
        "sessions": {"only": {"fingerprint": {}, "first_seen": "2026-10-26T00:00:00Z"}},
        "sections": {"vector": {"session": "only"}, "graph": {}},
    }
    primary, mismatches = sp.verify(ledger)
    assert primary == "only"
    assert mismatches == []


def test_stamp_is_idempotent_for_the_same_section(tmp_path, monkeypatch):
    ledger = tmp_path / "session.json"
    # Calibration probes are optional; skipping them keeps the test fast and
    # independent of numpy.
    monkeypatch.setattr(sp, "calibration", lambda: {"available": False})

    first = sp.stamp("vector", ["a.json"], path=str(ledger))
    second = sp.stamp("vector", ["b.json"], path=str(ledger))

    assert first["session"] == second["session"]
    assert second["artifacts"] == ["a.json", "b.json"]
    assert len(sp.load_ledger(str(ledger))["sessions"][second["session"]]["calibrations"]) == 2


def test_describe_calibration_pair_reports_first_and_last_with_drift():
    samples = [
        {"available": True, "random_gather_ns_per_access": 10.0, "triad_gbs": 20.0, "matmul_gflops": 100.0},
        {"available": True, "random_gather_ns_per_access": 9.0, "triad_gbs": 24.0, "matmul_gflops": 100.0},
    ]
    rows = dict((label, (before, after)) for label, before, after in sp.describe_calibration_pair(samples))

    gather = rows["random gather (ns/access)"]
    assert gather[0] == "10.0"
    assert gather[1].startswith("9.0 (-10.0%)")

    triad = rows["triad (GB/s)"]
    assert triad[1].startswith("24.0 (+20.0%)")


@pytest.mark.parametrize("value", [None, [], [{"available": False}]])
def test_describe_calibration_pair_handles_missing_samples(value):
    assert sp.describe_calibration_pair(value) == []
