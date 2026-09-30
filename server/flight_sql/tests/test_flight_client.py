import adbc_driver_flightsql.dbapi as flight_sql
import pandas as pd

print("Connecting to benostreamdb-flight at grpc://localhost:50051...")
with flight_sql.connect(uri="grpc://localhost:50051") as conn:
    print("Connection established!")

    with conn.cursor() as cur:
        # --- Catalog / namespace -------------------------------------------
        print("\nExecuting DDL: CREATE DATABASE / CREATE SCHEMA...")
        cur.execute("CREATE DATABASE IF NOT EXISTS mydb;")
        cur.execute("CREATE SCHEMA IF NOT EXISTS mydb.myschema;")

        # --- Table ----------------------------------------------------------
        print("\nExecuting DDL: CREATE TABLE mydb.myschema.test_table...")
        cur.execute("CREATE TABLE mydb.myschema.test_table (id INT, name VARCHAR);")

        print("Executing DML: INSERT INTO mydb.myschema.test_table...")
        cur.execute(
            "INSERT INTO mydb.myschema.test_table VALUES (1, 'Alice'), (2, 'Bob');"
        )

        # --- Indexes --------------------------------------------------------
        print("\nExecuting DDL: CREATE INDEX...")
        cur.execute("CREATE INDEX ON mydb.myschema.test_table (id);")

        # --- Primary key ----------------------------------------------------
        print("Executing DDL: ALTER TABLE ADD PRIMARY KEY...")
        cur.execute("ALTER TABLE mydb.myschema.test_table ADD PRIMARY KEY (id);")

        # --- Schema evolution ----------------------------------------------
        print("Executing DDL: ALTER TABLE ADD COLUMN...")
        cur.execute("ALTER TABLE mydb.myschema.test_table ADD COLUMN age INT;")

        # --- Maintenance ----------------------------------------------------
        print("Executing maintenance: OPTIMIZE / VACUUM / MSCK REPAIR...")
        cur.execute("OPTIMIZE TABLE mydb.myschema.test_table;")
        cur.execute("VACUUM mydb.myschema.test_table;")
        cur.execute("MSCK REPAIR TABLE mydb.myschema.test_table;")

        # --- Session settings ----------------------------------------------
        print("Executing SET / SHOW benostream.*...")
        cur.execute("SET benostream.warehouse = '/tmp/bsdb-warehouse';")
        cur.execute("SHOW benostream.warehouse;")
        print(f"warehouse = {cur.fetchall()}")

        # --- Metadata APIs --------------------------------------------------
        print("\nFetching metadata (Tables & Schemas)...")
        info = conn.adbc_get_objects(depth="all")
        tables = conn.adbc_get_table_schema("test_table")
        print(f"Table Schema for 'test_table': {tables}")

        # --- Query ----------------------------------------------------------
        print("\nExecuting Query: SELECT * FROM mydb.myschema.test_table...")
        cur.execute("SELECT * FROM mydb.myschema.test_table;")
        df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        print("Results:")
        print(df)

        # --- Information schema --------------------------------------------
        print("\nQuerying Information Schema...")
        try:
            cur.execute("SELECT * FROM information_schema.tables;")
            df_info = pd.DataFrame(
                cur.fetchall(), columns=[desc[0] for desc in cur.description]
            )
            print(df_info)
        except Exception as e:
            print(f"Failed to query information_schema: {e}")

        # --- Cleanup --------------------------------------------------------
        print("\nExecuting DDL: DROP TABLE...")
        cur.execute("DROP TABLE mydb.myschema.test_table;")
