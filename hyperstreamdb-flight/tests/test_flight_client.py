import adbc_driver_flightsql.dbapi as flight_sql
import pandas as pd

print("Connecting to hyperstreamdb-flight at grpc://localhost:50051...")
with flight_sql.connect(uri="grpc://localhost:50051") as conn:
    print("Connection established!")
    
    with conn.cursor() as cur:
        # Create a sample table
        print("\nExecuting DDL: CREATE TABLE test_table...")
        cur.execute("CREATE TABLE test_table (id INT, name VARCHAR);")
        
        print("Executing DML: INSERT INTO test_table...")
        cur.execute("INSERT INTO test_table VALUES (1, 'Alice'), (2, 'Bob');")
        
        # Test ADBC metadata APIs which map to Flight SQL do_get_*
        print("\nFetching metadata (Tables & Schemas)...")
        # ADBC mapping to GetTables/GetDbSchemas is via adbc_get_objects
        info = conn.adbc_get_objects(depth="all")
        # ADBC returns a stream of metadata which dbapi handles.
        # But wait, adbc_get_objects is low level. 
        # Using standard DBAPI to get tables
        tables = cur.adbc_get_table_schema(None, None, "test_table")
        print(f"Table Schema for 'test_table': {tables}")

        # Execute a query
        print("\nExecuting Query: SELECT * FROM test_table...")
        cur.execute("SELECT * FROM test_table;")
        df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        print("Results:")
        print(df)
        
        # Test Information Schema
        print("\nQuerying Information Schema...")
        try:
            cur.execute("SELECT * FROM information_schema.tables;")
            df_info = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
            print(df_info)
        except Exception as e:
            print(f"Failed to query information_schema: {e}")
