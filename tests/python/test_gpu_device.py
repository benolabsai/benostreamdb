# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

"""GPU device selection is exposed on every surface via one shared mapping.

`benostreamdb.set_gpu_device` / `gpu_device` are the Python entry points; the
Spark/Trino JNI bridges, the Flight server, and the MCP server all resolve a
device string through the same core `context_from_device_str`.
"""

import benostreamdb

_BACKENDS = {"cpu", "cuda", "mps", "intel", "rocm"}


def test_gpu_device_cpu_roundtrip():
    assert benostreamdb.set_gpu_device("cpu") == "cpu"
    assert benostreamdb.gpu_device() == "cpu"


def test_gpu_device_auto_resolves_to_a_backend():
    resolved = benostreamdb.set_gpu_device("auto")
    assert resolved in _BACKENDS
    assert benostreamdb.gpu_device() == resolved


def test_gpu_device_unknown_falls_back_to_auto():
    # Unknown input must not error; it falls back to auto-detection.
    resolved = benostreamdb.set_gpu_device("not-a-real-device")
    assert resolved in _BACKENDS
