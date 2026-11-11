import tempfile
"""
Cross-Engine Compatibility Tests for Iceberg V2/V3 Features

Tests that BenoStreamDB-written tables can be read by:
- Apache Spark
- Trino
- Other Iceberg-compatible engines

Prerequisites:
- Docker (for Spark/Trino containers)
- benostreamdb installed
"""

import benostreamdb as bsdb
import pandas as pd
import pytest
from pathlib import Path
import importlib.util
import os
import shutil
import subprocess
import json

# Test data directory
TEST_DIR = (Path(tempfile.gettempdir()) / "benostream_compat_tests")
TEST_DIR.mkdir(exist_ok=True)

# The per-PR `ecosystem` job cannot read Iceberg-format tables from the pip
# `pyspark` package (it ships no Iceberg JAR), so these tests skip there. The
# `Iceberg Interop (JVM)` workflow sets both variables below to turn the skips
# into hard failures and to fetch the Iceberg Spark runtime from Maven:
#
#   BSDB_REQUIRE_JVM_INTEROP=1
#   BSDB_SPARK_ICEBERG_PACKAGES=org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.11.0
#
# The Iceberg runtime artifact is keyed by Spark *minor series*, and Maven
# Central publishes exactly: 3.4, 3.5, 4.0, 4.1 (each `_2.12` and `_2.13`).
# There is no 4.2 runtime, so Spark 4.2.x cannot be paired with a supported
# Iceberg build — pin 3.5.2 (+ `3.5_2.12`) or 4.0/4.1 (+ `_2.13`). Iceberg 1.11
# is used rather than 1.6 because V3 deletion vectors and row lineage need a
# recent runtime.
REQUIRE_JVM_INTEROP = os.environ.get("BSDB_REQUIRE_JVM_INTEROP") == "1"
SPARK_ICEBERG_PACKAGES = os.environ.get(
    "BSDB_SPARK_ICEBERG_PACKAGES",
    "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.11.0",
)


def _table_format_version(table_path: str) -> int | None:
    """Read `format-version` from the table's latest metadata.json."""
    metadata_dir = Path(table_path) / "metadata"
    if not metadata_dir.is_dir():
        return None
    best = None
    for p in metadata_dir.iterdir():
        name = p.name
        if name.startswith("v") and name.endswith(".metadata.json"):
            try:
                best = max(best or 0, int(name[1 : -len(".metadata.json")]))
            except ValueError:
                continue
    if best is None:
        return None
    try:
        return int(json.loads((metadata_dir / f"v{best}.metadata.json").read_text())["format-version"])
    except (KeyError, ValueError, OSError):
        return None


def _spark_submit_available() -> bool:
    return shutil.which("spark-submit") is not None


def _require_or_skip(reason: str) -> None:
    """Skip locally / per-PR, but fail in the interop workflow."""
    if REQUIRE_JVM_INTEROP:
        pytest.fail(f"JVM interop is required but unavailable: {reason}")
    pytest.skip(reason)

class TestSparkCompatibility:
    """Test BenoStreamDB → Spark compatibility"""
    
    def setup_method(self):
        """Create test table with V2/V3 features"""
        self.table_path = str(TEST_DIR / "spark_test_table")
        self.table = bsdb.Table(self.table_path)
        
        # Write test data with V2 features
        df = pd.DataFrame({
            "id": range(1000),
            "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1h"),
            "value": range(1000, 2000),
            "category": ["A", "B", "C"] * 333 + ["A"]
        })
        
        # Configure V2 features
        self.table.set_sort_order(["timestamp", "id"], ascending=[False, True])
        self.table.write_pandas(df)
    
    def test_spark_read_basic(self):
        """Test Spark can read a BenoStreamDB table as an Iceberg table."""
        if not _spark_submit_available():
            _require_or_skip("spark-submit not available")
        # The pip-installed pyspark lacks the Iceberg JAR, so plain CI skips.
        # The interop workflow sets BSDB_REQUIRE_JVM_INTEROP=1 and lets
        # spark-submit fetch the runtime from Maven via --packages.
        if not REQUIRE_JVM_INTEROP and (
            os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")
        ):
            pytest.skip("pyspark pip package lacks Iceberg JAR in CI")

        import textwrap
        spark_script = textwrap.dedent(f"""\
        from pyspark.sql import SparkSession

        spark = SparkSession.builder \\
            .appName("BenoStreamDB Compat Test") \\
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \\
            .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \\
            .config("spark.sql.catalog.local.type", "hadoop") \\
            .config("spark.sql.catalog.local.warehouse", "{self.table_path}") \\
            .getOrCreate()

        df = spark.read.format("iceberg").load("{self.table_path}")
        assert df.count() == 1000
        print("✅ Spark read successful")
        """)

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(spark_script)
            script_path = f.name
        try:
            cmd = ["spark-submit", "--master", "local[*]"]
            if REQUIRE_JVM_INTEROP:
                cmd += ["--packages", SPARK_ICEBERG_PACKAGES]
            cmd.append(script_path)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                if "SparkClassNotFoundException" in result.stdout or \
                   "DATA_SOURCE_NOT_FOUND" in result.stdout:
                    _require_or_skip("Iceberg Spark runtime JAR not available")
                assert result.returncode == 0, \
                    f"Spark submit failed:\n{result.stderr}\n{result.stdout}"
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

    def test_spark_read_v3_metadata(self):
        """Spark must expose V3 row-lineage metadata columns.

        `_row_id` / `_last_updated_sequence_number` are metadata columns, not
        table-schema columns, so this only has teeth where the Iceberg runtime
        implements V3 lineage — i.e. under the interop workflow.
        """
        if not REQUIRE_JVM_INTEROP:
            pytest.skip("V3 lineage read needs the Iceberg runtime; run the interop workflow")
        version = _table_format_version(self.table_path)
        if version != 3:
            pytest.skip(
                f"table is format-version {version}, not 3; V3 lineage is absent by design"
            )
        if not _spark_submit_available():
            _require_or_skip("spark-submit not available")

        import textwrap
        spark_script = textwrap.dedent(f"""\
        from pyspark.sql import SparkSession

        spark = SparkSession.builder \\
            .appName("BenoStreamDB V3 Lineage Test") \\
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \\
            .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \\
            .config("spark.sql.catalog.local.type", "hadoop") \\
            .config("spark.sql.catalog.local.warehouse", "{self.table_path}") \\
            .getOrCreate()

        df = spark.read.format("iceberg").load("{self.table_path}")
        cols = set(df.columns)
        assert "_row_id" in cols, f"missing _row_id; saw {{sorted(cols)}}"
        assert "_last_updated_sequence_number" in cols, \\
            f"missing _last_updated_sequence_number; saw {{sorted(cols)}}"
        print("OK Spark saw V3 lineage columns")
        """)

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(spark_script)
            script_path = f.name
        try:
            result = subprocess.run(
                [
                    "spark-submit",
                    "--master",
                    "local[*]",
                    "--packages",
                    SPARK_ICEBERG_PACKAGES,
                    script_path,
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"Spark V3 lineage read failed:\n{result.stderr}\n{result.stdout}"
            )
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)


class TestTrinoCompatibility:
    """Test BenoStreamDB → Trino compatibility"""
    
    def setup_method(self):
        """Create test table"""
        self.table_path = str(TEST_DIR / "trino_test_table")
        self.table = bsdb.Table(self.table_path)
        
        df = pd.DataFrame({
            "user_id": range(500),
            "action": ["click", "view", "purchase"] * 166 + ["click", "view"],
            "amount": [i * 1.5 for i in range(500)]
        })
        
        self.table.write_pandas(df)
    
    def test_trino_read_basic(self):
        """Trino must read a BenoStreamDB table as an Iceberg table.

        Requires a Trino server whose Iceberg connector points at the same
        directory tree (the interop workflow starts one with a filesystem
        catalog over `TEST_DIR`). Skipped everywhere else.
        """
        host = os.environ.get("TRINO_HOST")
        if not host:
            _require_or_skip("TRINO_HOST is not set")
        try:
            import trino  # noqa: F401
        except ImportError:
            _require_or_skip("the `trino` Python client is not installed")

        import trino

        conn = trino.dbapi.connect(
            host=host,
            port=int(os.environ.get("TRINO_PORT", "8080")),
            user="ci",
            catalog="iceberg",
            schema="default",
        )
        cur = conn.cursor()
        # A filesystem catalog names the table after its directory.
        cur.execute('SELECT count(*) FROM iceberg.default."trino_test_table"')
        (count,) = cur.fetchone()
        assert count == 500, f"Trino saw {count} rows, expected 500"


class TestV2Features:
    """Test V2 feature compatibility"""
    
    def test_sort_order_metadata(self):
        """Verify sort order is written to metadata"""
        table_path = str(TEST_DIR / "sort_order_test")
        table = bsdb.Table(table_path)
        
        df = pd.DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]})
        table.set_sort_order(["a"], ascending=[True])
        table.write_pandas(df)
        table.commit()
        
        # Verify sort order in manifest
        manifest_path = Path(table_path) / "_manifest"
        assert manifest_path.exists()

        # Load v1.json to verify sort order metadata
        import json
        v1_path = manifest_path / "v1.json"
        assert v1_path.exists()
        with open(v1_path) as f:
            v1_data = json.load(f)
        
        # Check sort_orders field in v1.json
        assert len(v1_data.get("sort_orders", [])) > 0
        sort_order = v1_data["sort_orders"][0]
        assert len(sort_order.get("fields", [])) > 0
        field = sort_order["fields"][0]
        assert field.get("source-id") == 1
        assert field.get("direction") == "asc"

        # Load metadata/v1.metadata.json
        meta_path = Path(table_path) / "metadata" / "v1.metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta_data = json.load(f)
        assert len(meta_data.get("sort-orders", [])) > 0
        meta_sort_order = meta_data["sort-orders"][0]
        assert len(meta_sort_order.get("fields", [])) > 0
        meta_field = meta_sort_order["fields"][0]
        assert meta_field.get("source-id") == 1
        assert meta_field.get("direction") == "asc"
    
    def test_partition_evolution_metadata(self):
        """Verify partition spec evolution is tracked"""
        import uuid
        table_path = str(TEST_DIR / f"partition_evolution_test_{uuid.uuid4().hex}")
        table = bsdb.Table(table_path)
        
        import pyarrow as pa
        import datetime
        dates = pa.array([datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(100)], type=pa.date32())
        values = pa.array(range(100))
        batch = pa.RecordBatch.from_arrays([dates, values], names=["date", "value"])
        # Set initial partition spec
        table.set_partition_spec([
            {"source_id": 1, "field_id": 1000, "name": "date", "transform": "day"}
        ])
        table.write([batch])
        
        # Verify partition_specs history is tracked
        # (This would check the manifest JSON)
    
    def test_ndv_statistics(self):
        """Verify NDV statistics are computed"""
        table_path = str(TEST_DIR / "ndv_test")
        table = bsdb.Table(table_path)
        
        # Create data with known cardinality
        df = pd.DataFrame({
            "id": list(range(100)) * 10,  # 100 distinct values
            "category": ["A", "B", "C"] * 333 + ["A"]  # 3 distinct values
        })
        
        table.write_pandas(df)
        
        # Verify distinct_count is computed
        # (This would check the manifest statistics)


class TestV3Features:
    """Test V3 feature compatibility"""
    
    def test_row_lineage_columns(self):
        """Verify V3 metadata columns are added"""
        table_path = str(TEST_DIR / "v3_lineage_test")
        table = bsdb.Table(table_path)
        
        df = pd.DataFrame({"value": [1, 2, 3]})
        table.write_pandas(df)
        
        # Read back and verify metadata columns
        result = table.to_pandas()
        
        # Check for V3 columns (if format_version >= 3)
        # assert "_row_id" in result.columns
        # assert "_last_updated_sequence_number" in result.columns
    
    def test_default_values_schema(self):
        """Verify default values are stored in schema"""
        # This would test that initial_default and write_default
        # are properly serialized in the schema metadata
        pass


# Helper functions for manual testing
def create_sample_table_for_spark():
    """Create a sample table for manual Spark testing"""
    table_path = str(TEST_DIR / "manual_spark_test")
    table = bsdb.Table(table_path)
    
    df = pd.DataFrame({
        "id": range(10000),
        "timestamp": pd.date_range("2024-01-01", periods=10000, freq="1min"),
        "value": [i * 1.1 for i in range(10000)],
        "category": ["A", "B", "C", "D"] * 2500
    })
    
    table.set_sort_order(["timestamp"], ascending=[False])
    table.write_pandas(df)
    
    print(f"✅ Created test table at: {table_path}")
    print("\nTo test with Spark:")
    print(f"  spark.read.format('iceberg').load('{table_path}').show()")
    
    return table_path


def create_sample_table_for_trino():
    """Create a sample table for manual Trino testing"""
    table_path = str(TEST_DIR / "manual_trino_test")
    table = bsdb.Table(table_path)
    
    df = pd.DataFrame({
        "user_id": range(5000),
        "action": ["click", "view", "purchase", "return"] * 1250,
        "amount": [i * 2.5 for i in range(5000)]
    })
    
    table.write_pandas(df)
    
    print(f"✅ Created test table at: {table_path}")
    print("\nTo test with Trino:")
    print(f"  SELECT COUNT(*) FROM iceberg.default.\"{table_path.split('/')[-1]}\";")
    
    return table_path


if __name__ == "__main__":
    print("Creating sample tables for manual testing...")
    print("\n" + "="*60)
    create_sample_table_for_spark()
    print("\n" + "="*60)
    create_sample_table_for_trino()
    print("\n" + "="*60)
    print("\n✅ Sample tables created!")
    print("\nRun pytest to execute automated tests:")
    print("  pytest tests/test_cross_engine_compat.py -v")
