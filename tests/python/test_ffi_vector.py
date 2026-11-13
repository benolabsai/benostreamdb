# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

"""Manual FFI probe for the JNI bridge cdylib.

The JNI entry points live in `libbenostreamdb.so`, which is produced by a local
`cargo build` and is **not** part of the Python wheel. CI installs the wheel, so
this only runs where that artifact exists; the load is guarded accordingly
instead of failing the unit shard with a missing shared object.

The call itself is not exercised: `Java_..._vectorSearch` decodes its arguments
through a real `JNIEnv`, so a Python-side ctypes call would need a C wrapper.
Rust-side coverage for the same conversion lives in the crate's own tests.
"""

import ctypes
from pathlib import Path

import pytest

LIB_PATH = Path("target/debug/libbenostreamdb.so")


@pytest.mark.skipif(
    not LIB_PATH.exists(),
    reason=f"{LIB_PATH} is a local cargo build artifact, not part of the wheel",
)
def test_ffi_vector():
    lib = ctypes.CDLL(str(LIB_PATH))
    assert lib is not None


if __name__ == "__main__":
    print("Test ready.")
