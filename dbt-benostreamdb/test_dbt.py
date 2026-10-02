import sys
try:
    from dbt.adapters.benostreamdb.connections import BenoStreamDBConnectionManager, BenoStreamDBCredentials
    import benostreamdb

    session = benostreamdb.Session()
    result = session.sql("SELECT 1 AS id")
    print(result.to_pandas())
    print("Success")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
