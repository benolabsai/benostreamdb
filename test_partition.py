import hyperstreamdb as hdb
import pyarrow as pa
import os, shutil
import pandas as pd

if os.path.exists('/tmp/part_demo'):
    shutil.rmtree('/tmp/part_demo')

schema = pa.schema([
    ('id', pa.int32()), 
    ('category', pa.string())
])

spec = {
    'fields': [
        {'name': 'category', 'transform': 'identity', 'source_id': 1, 'field_id': 1000}
    ]
}

table = hdb.Table.create_partitioned('file:///tmp/part_demo', schema, spec)
table.write(pd.DataFrame({'id': [1, 2], 'category': ['A', 'B']}))
table.commit()

entries = table.manifest().entries
print(f'Entries: {len(entries)}')
for entry in entries:
    print(f' - Path: {entry.file_path}')
    print(f' - Partition Values: {entry.partition_values}')
