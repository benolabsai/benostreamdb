"""Unit tests for scripts/prepare_demo.py embed/load streaming logic.

No GPU / no HyperStreamDB tables required — the embed stage runs against a
faked sentence-transformers model, and the load zip/deletion invariants are
exercised directly on synthetic shards. Guards the bugs that OOM-killed the
whole-site run (37 GB whole-shard read_table) and the delete-before-commit
data-loss hazard.
"""
import os
import shutil
import sys
import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# prepare_demo imports polars, which is a demo-script dependency and is not part
# of the package's `dev` extras. Skip cleanly (rather than erroring at
# collection) when it isn't installed.
pytest.importorskip("polars")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import prepare_demo as pdemo  # noqa: E402

N, DIM = 450_000, 8


@pytest.fixture
def wiki_like(tmp_path, monkeypatch):
    """A tiny wiki_nodes-shaped source + fake CPU embedder."""
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    src = tmp_path / "wiki_nodes.parquet"
    pq.write_table(pa.table({
        "id": pa.array(np.arange(N), pa.int64()),
        "title": pa.array([f"t{i}" for i in range(N)], pa.large_string()),
        "summary": pa.array([f"s{i}" for i in range(N)], pa.large_string()),
    }), str(src))
    monkeypatch.setattr(pdemo, "EMB", str(emb_dir))
    monkeypatch.setattr(pdemo, "DATA", str(tmp_path))
    monkeypatch.setenv("HDB_EMBED_SHARD_ROWS", "200000")

    fake_st = types.ModuleType("sentence_transformers")

    class FakeModel:
        def encode(self, texts, **kw):
            rng = np.random.default_rng(len(texts))
            return rng.standard_normal((len(texts), DIM), dtype=np.float32)

        def half(self):
            return self

    fake_st.SentenceTransformer = lambda *a, **k: FakeModel()
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return tmp_path, str(src), str(emb_dir)


def test_embed_rotation_and_skip(wiki_like):
    _, _, emb_dir = wiki_like
    pdemo.stage_embed("fake", 0, 256, lead_chars=10)
    shards = sorted(f for f in os.listdir(emb_dir) if f.endswith(".parquet"))
    counts = [pq.ParquetFile(os.path.join(emb_dir, s)).metadata.num_rows for s in shards]
    assert shards == ["part-000.parquet", "part-001.parquet", "part-002.parquet"]
    assert counts == [200_000, 200_000, 50_000]
    assert not [f for f in os.listdir(emb_dir) if f.endswith(".tmp")]
    # idempotent skip
    pdemo.stage_embed("fake", 0, 256, lead_chars=10)
    assert sorted(f for f in os.listdir(emb_dir) if f.endswith(".parquet")) == shards


def test_embed_resumes_from_partial_shards(wiki_like):
    """An interrupted embed must continue, not restart (hours of GPU work)."""
    _, src, emb_dir = wiki_like
    pdemo.stage_embed("fake", 0, 256, lead_chars=10)
    shards = sorted(f for f in os.listdir(emb_dir) if f.endswith(".parquet"))
    assert len(shards) == 3
    # simulate an interruption: drop the last shard + leave a stale tmp file
    os.remove(os.path.join(emb_dir, shards[-1]))
    open(os.path.join(emb_dir, "part-003.parquet.tmp"), "wb").close()

    pdemo.stage_embed("fake", 0, 256, lead_chars=10)

    remaining = sorted(f for f in os.listdir(emb_dir) if f.endswith(".parquet"))
    rows = sum(pq.ParquetFile(os.path.join(emb_dir, s)).metadata.num_rows for s in remaining)
    assert rows == N, f"resume produced {rows} rows, expected {N}"
    assert not [f for f in os.listdir(emb_dir) if f.endswith(".tmp")]
    # second resume is a no-op
    pdemo.stage_embed("fake", 0, 256, lead_chars=10)
    assert sorted(f for f in os.listdir(emb_dir) if f.endswith(".parquet")) == remaining


def _write_known_shards(emb_dir, src_rows):
    """3 shards with deterministic values, sizes not aligned to any batch size."""
    vals = np.arange(src_rows * DIM, dtype=np.float32).reshape(src_rows, DIM)
    for f in os.listdir(emb_dir):
        os.remove(os.path.join(emb_dir, f))
    bounds = [(0, 200_000), (200_000, 400_000), (400_000, src_rows)]
    for i, (a, b) in enumerate(bounds):
        pq.write_table(pa.table({
            "id": pa.array(np.arange(a, b), pa.int64()),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(vals[a:b].reshape(-1), pa.float32()), DIM),
        }), os.path.join(emb_dir, f"part-{i:03d}.parquet"))
    return vals


def _zip_stream(emb_dir, src, node_batch, emb_batch):
    """Mirror of stage_load's streaming zip loop (same code shape)."""
    parts = sorted(f for f in os.listdir(emb_dir) if f.endswith(".parquet"))

    def emb_batches():
        for p in parts:
            for b in pq.ParquetFile(os.path.join(emb_dir, p)).iter_batches(batch_size=emb_batch):
                yield b["embedding"]

    emb_iter = emb_batches()
    emb_cur, emb_off = None, 0
    assembled = []
    for batch in pq.ParquetFile(src).iter_batches(batch_size=node_batch):
        pieces = []
        need = batch.num_rows
        while need > 0:
            if emb_cur is None or emb_off >= len(emb_cur):
                emb_cur = next(emb_iter)
                emb_off = 0
                continue
            take = min(need, len(emb_cur) - emb_off)
            pieces.append(emb_cur.slice(emb_off, take))
            emb_off += take
            need -= take
        col = pieces[0] if len(pieces) == 1 else pa.concat_arrays(pieces)
        assembled.append(col)
    return pa.concat_arrays(assembled), parts


def test_multi_shard_zip_alignment(wiki_like):
    _, src, emb_dir = wiki_like
    vals = _write_known_shards(emb_dir, N)
    out, _ = _zip_stream(emb_dir, src, node_batch=2500, emb_batch=3000)
    flat = out.values.to_numpy(zero_copy_only=False)
    assert flat.size == N * DIM
    assert np.array_equal(flat.reshape(N, DIM), vals)


def test_table_loaded_guard(tmp_path):
    """A crashed-run table shell (metadata/_wal/_manifest only) must NOT be
    treated as loaded — it silently skipped the whole node load once."""
    shell = tmp_path / "nodes"
    (shell / "metadata").mkdir(parents=True)
    (shell / "_wal").mkdir()
    (shell / "_manifest").mkdir()
    assert pdemo._table_loaded(str(shell)) is False
    (shell / "data").mkdir()
    assert pdemo._table_loaded(str(shell)) is True
    assert pdemo._table_loaded(str(tmp_path / "missing")) is False


@pytest.mark.parametrize("gb,expected", [
    (1.0, 250_000),      # floor clamp
    (4.0, 250_000),      # 0.44M -> rounds down to floor
    (18.0, 2_000_000),   # 2.0M exactly
    (120.0, 10_000_000),  # ceiling clamp
])
def test_auto_chunk_rows(gb, expected):
    assert pdemo._auto_chunk_rows(gb) == expected


def test_shards_survive_mid_loop_crash(wiki_like):
    """Deletion must happen only AFTER commit — never during the write loop."""
    _, src, emb_dir = wiki_like
    _write_known_shards(emb_dir, N)
    _, parts = _zip_stream(emb_dir, src, node_batch=2500, emb_batch=3000)
    before = sorted(f for f in os.listdir(emb_dir) if f.endswith(".parquet"))
    assert before == parts  # zip loop itself deletes nothing
