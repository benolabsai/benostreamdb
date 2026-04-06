import unittest
import numpy as np
import hyperstreamdb as hdb
import pyarrow as pa
import os
import shutil

class TestPgVectorCompatibility(unittest.TestCase):
    def setUp(self):
        self.db_path = "/tmp/hdb_pgvector_test"
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        
        # Create a test table
        schema = pa.schema([
            ("id", pa.int32()),
            ("category", pa.string()),
            ("embedding", pa.list_(pa.float32(), 3))
        ])
        
        data = [
            [1, 2, 3],
            ["A", "A", "B"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ]
        table = pa.Table.from_arrays(data, schema=schema)
        # In 0.1.9, we register tables by wrapping them in HyperStreamTable
        # For this test, we rewrite the test setup to use a REAL local table
        # instead of an in-memory arrow table to stay compatible with the registration logic.
        hdb_table = hdb.Table(self.db_path)
        hdb_table.write(table)
        
        self.session = hdb.Session()
        self.session.register("news", hdb_table)

    def tearDown(self):
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

    def test_pgvector_operator_l2(self):
        # Test <-> operator with string literal
        query = "SELECT id, embedding <-> '[1.0, 0.0, 0.0]' AS dist FROM news ORDER BY dist LIMIT 1"
        df = self.session.sql(query).to_pandas()
        self.assertEqual(df.iloc[0]['id'], 1)
        self.assertAlmostEqual(df.iloc[0]['dist'], 0.0, places=5)

    def test_pgvector_cast_syntax(self):
        # Test ::vector cast
        query = "SELECT id, embedding <-> '[0.0, 1.0, 0.0]'::vector AS dist FROM news ORDER BY dist LIMIT 1"
        df = self.session.sql(query).to_pandas()
        self.assertEqual(df.iloc[0]['id'], 2)

    def test_numpy_scalar_interpolation(self):
        # Test NumPy scalars
        vec = [np.float32(0.0), np.float32(0.0), np.float32(1.0)]
        vec_str = str(vec).replace(' ', '')
        
        query = f"SELECT id, embedding <-> '{vec_str}' AS dist FROM news ORDER BY dist LIMIT 1"
        df = self.session.sql(query).to_pandas()
        self.assertEqual(df.iloc[0]['id'], 3)

    def test_large_vector_string(self):
        # Test large vector string
        long_val = "0.0000000000000000000000000000000000000000001"
        query = f"SELECT id, embedding <-> '[1.0, 0.0, {long_val}]' AS dist FROM news"
        df = self.session.sql(query).to_pandas()
        self.assertGreater(len(df), 0)

    def test_vector_avg_centroid(self):
        # Test vector_avg
        query = "SELECT vector_avg(embedding) FROM news"
        df = self.session.sql(query).to_pandas()
        # [1/3, 1/3, 1/3]
        avg_vec = df.iloc[0, 0]
        self.assertAlmostEqual(avg_vec[0], 0.33333334, places=6)

    def test_mixed_aggregations(self):
        # Test vector_avg(embedding) AND standard aggregator COUNT(*)
        query = """
            SELECT category, 
                   vector_avg(embedding) AS centroid, 
                   COUNT(*) as count
            FROM news 
            GROUP BY category
            ORDER BY category
        """
        df = self.session.sql(query).to_pandas()
        
        # Category A has 2 items, Category B has 1 item
        self.assertEqual(len(df), 2)
        
        row_a = df[df['category'] == 'A'].iloc[0]
        self.assertEqual(row_a['count'], 2)
        # Average of [1,0,0] and [0,1,0] is [0.5, 0.5, 0]
        self.assertAlmostEqual(row_a['centroid'][0], 0.5, places=5)
        self.assertAlmostEqual(row_a['centroid'][1], 0.5, places=5)
        
        row_b = df[df['category'] == 'B'].iloc[0]
        self.assertEqual(row_b['count'], 1)
        # Only [0,0,1]
        self.assertAlmostEqual(row_b['centroid'][2], 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
