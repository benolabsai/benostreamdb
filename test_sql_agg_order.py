import hyperstreamdb as hdb
import pyarrow as pa
import numpy as np
import pandas as pd
import os, shutil

uri = 'file:///tmp/test_agg_order'
if os.path.exists('/tmp/test_agg_order'):
    shutil.rmtree('/tmp/test_agg_order')

schema = pa.schema([
    ('category', pa.string()), 
    ('embedding', pa.list_(pa.float32(), 3))
])

table = hdb.Table.create(uri, schema)
table.write(pd.DataFrame({
    'category': ['A', 'A', 'B'], 
    'embedding': [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
}))
table.commit()

session = hdb.Session()
session.register('news', table)

query = """
    SELECT category, 
           vector_avg(embedding) AS centroid,
           COUNT(*) as count
    FROM news
    GROUP BY category
    ORDER BY count DESC
"""

try:
    df = session.sql(query).to_pandas()
    print('Success')
    print(df)
except Exception as e:
    print(f'ERROR: {e}')
