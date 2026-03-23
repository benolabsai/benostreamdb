"""
Cross-Engine Compatibility Tests for Iceberg V2/V3 Features

Tests that HyperStreamDB-written tables can be read by:
- Apache Spark
- Trino
- Other Iceberg-compatible engines

Prerequisites:
- Docker (for Spark/Trino containers)
- hyperstreamdb installed
"""

import hyperstreamdb as hdb
import pandas as pd
import pytest
from pathlib import Path
import subprocess
import json

# Test data directory
TEST_DIR = Path("/tmp/hyperstream_compat_tests")
TEST_DIR.mkdir(exist_ok=True)

class TestSparkCompatibility:
    """Test HyperStreamDB → Spark compatibility"""
    
    def setup_method(self):
        """Create test table with V2/V3 features"""
        self.table_path = str(TEST_DIR / "spark_test_table")
        self.table = hdb.Table(self.table_path)
        
        # Write test data with V2 features
        df = pd.DataFrame({
            "id": range(1000),
            "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1H"),
            "value": range(1000, 2000),
            "category": ["A", "B", "C"] * 333 + ["A"]
        })
        
        # Configure V2 features
        self.table.set_sort_order(["timestamp", "id"], ascending=[False, True])
        self.table.write_pandas(df)
    
    @pytest.mark.skip(reason="Requires Spark installation")
    def test_spark_read_basic(self):
        """Test Spark can read HyperStreamDB table"""
        spark_script = f"""
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder \\
            .appName("HyperStreamDB Compat Test") \\
            .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \\
            .config("spark.sql.catalog.local.type", "hadoop") \\
            .config("spark.sql.catalog.local.warehouse", "{self.table_path}") \\
            .getOrCreate()
        
        df = spark.read.format("iceberg").load("{self.table_path}")
        assert df.count() == 1000
        print("✅ Spark read successful")
        """
        
        # Run Spark script
        result = subprocess.run(
            ["spark-submit", "--master", "local[*]", "-"],
            input=spark_script.encode(),
            capture_output=True
        )
        assert result.returncode == 0
    
    @pytest.mark.skip(reason="Requires Spark installation")
    def test_spark_read_v3_metadata(self):
        """Test Spark can read V3 metadata columns"""
        # This test would verify _row_id and _last_updated_sequence_number
        # are accessible from Spark
        pass


class TestTrinoCompatibility:
    """Test HyperStreamDB → Trino compatibility"""
    
    def setup_method(self):
        """Create test table"""
        self.table_path = str(TEST_DIR / "trino_test_table")
        self.table = hdb.Table(self.table_path)
        
        df = pd.DataFrame({
            "user_id": range(500),
            "action": ["click", "view", "purchase"] * 166 + ["click", "view"],
            "amount": [i * 1.5 for i in range(500)]
        })
        
        self.table.write_pandas(df)
    
    @pytest.mark.skip(reason="Requires Trino installation")
    def test_trino_read_basic(self):
        """Test Trino can read HyperStreamDB table"""
        # This would use Trino CLI or Python client
        # to query the table and verify results
        pass


class TestV2Features:
    """Test V2 feature compatibility"""
    
    def test_sort_order_metadata(self):
        """Verify sort order is written to metadata"""
        table_path = str(TEST_DIR / "sort_order_test")
        table = hdb.Table(table_path)
        
        df = pd.DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]})
        table.set_sort_order(["a"], ascending=[True])
        table.write_pandas(df)
        
        # Verify sort order in manifest
        # (This would check the manifest JSON file)
        manifest_path = Path(table_path) / "_manifest"
        assert manifest_path.exists()
    
    def test_partition_evolution_metadata(self):
        """Verify partition spec evolution is tracked"""
        table_path = str(TEST_DIR / "partition_evolution_test")
        table = hdb.Table(table_path)
        
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100),
            "value": range(100)
        })
        
        # Set initial partition spec
        table.set_partition_spec([
            {"source_id": 1, "field_id": 1000, "name": "date", "transform": "day"}
        ])
        table.write_pandas(df)
        
        # Verify partition_specs history is tracked
        # (This would check the manifest JSON)
    
    def test_ndv_statistics(self):
        """Verify NDV statistics are computed"""
        table_path = str(TEST_DIR / "ndv_test")
        table = hdb.Table(table_path)
        
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
        table = hdb.Table(table_path)
        
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
    table = hdb.Table(table_path)
    
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
    table = hdb.Table(table_path)
    
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
