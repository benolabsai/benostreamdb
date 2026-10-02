"""Loader-safe GPU device selection.

A broken or version-conflicting CUDA/nvrtc library can crash the *dynamic
linker* (SIGSEGV in ``_dl_init`` / ``_dl_runtime_resolve``), which no in-process
``try``/``except`` can catch. These tests assert that device selection probes
GPU backends in a throwaway subprocess and degrades to CPU instead of killing
the process.
"""

import benostreamdb
from benostreamdb import Device


def test_cpu_device_always_works():
    assert Device("cpu") is not None


def test_probe_returns_bool_and_never_crashes():
    ok = benostreamdb._probe_backend_subprocess("cuda")
    assert isinstance(ok, bool)


def test_probe_short_circuits_inside_child(monkeypatch):
    # Inside a probe child the guard must return True without spawning again
    # (otherwise the probe would recurse forever).
    monkeypatch.setenv("BSDB_SKIP_GPU_PROBE", "1")
    assert benostreamdb._probe_backend_subprocess("cuda") is True


def test_cuda_device_never_crashes():
    # Either a real CUDA device or a CPU fallback — never a crash.
    d = Device("cuda")
    assert d is not None
    assert getattr(d, "type_name", "cpu") in ("cuda", "rocm", "cpu")
