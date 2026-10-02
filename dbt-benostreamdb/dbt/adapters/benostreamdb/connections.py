from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, Any

from dbt.adapters.contracts.connection import Credentials, Connection, AdapterResponse
from dbt.adapters.sql import SQLConnectionManager
from dbt.exceptions import DbtRuntimeError
from dbt.adapters.events.logging import AdapterLogger

import benostreamdb

logger = AdapterLogger("BenoStreamDB")


@dataclass
class BenoStreamDBCredentials(Credentials):
    path: str = ":memory:"
    database: Optional[str] = "benostreamdb"
    schema: str = "public"

    @property
    def type(self):
        return "benostreamdb"

    @property
    def unique_field(self):
        return self.path

    def _connection_keys(self):
        return ("path", "database", "schema")


class BenoStreamDBCursor:
    def __init__(self, session):
        self.session = session
        self.description = None
        self._results = []

    def execute(self, sql: str, bindings=None):
        if bindings:
            # Note: dbt-core handles most string formatting. For complex bindings, DBAPI compliance would be needed.
            pass
        
        try:
            arrow_table = self.session.sql(sql)
            if arrow_table is not None and hasattr(arrow_table, "schema"):
                self.description = [(f.name, f.type) for f in arrow_table.schema]
                df = arrow_table.to_pandas()
                self._results = [tuple(x) for x in df.to_numpy()]
            else:
                self.description = None
                self._results = []
        except Exception as e:
            raise Exception(f"Error executing {sql}: {e}")
            
    def fetchall(self):
        return self._results

    def fetchone(self):
        if self._results:
            return self._results.pop(0)
        return None

    def close(self):
        pass


class BenoStreamDBConnectionWrapper:
    def __init__(self, session):
        self.session = session
        
    def cursor(self):
        return BenoStreamDBCursor(self.session)
        
    def commit(self):
        pass
        
    def rollback(self):
        pass
        
    def close(self):
        pass


class BenoStreamDBConnectionManager(SQLConnectionManager):
    TYPE = "benostreamdb"
    _session = None

    @classmethod
    def get_session(cls, credentials):
        if cls._session is None:
            # Instantiate a single embedded engine session for all connections
            cls._session = benostreamdb.Session()
        return cls._session

    @classmethod
    def open(cls, connection: Connection) -> Connection:
        if connection.state == "open":
            logger.debug("Connection is already open, skipping open.")
            return connection

        credentials = connection.credentials
        
        try:
            session = cls.get_session(credentials)
            connection.handle = BenoStreamDBConnectionWrapper(session)
            connection.state = "open"
        except Exception as e:
            logger.error(f"Error initializing BenoStreamDB embedded engine: {e}")
            connection.handle = None
            connection.state = "fail"
            raise DbtRuntimeError(f"Failed to connect to BenoStreamDB: {e}")

        return connection

    @classmethod
    def get_response(cls, cursor: Any) -> AdapterResponse:
        return AdapterResponse(_message="OK")

    def cancel(self, connection: Connection):
        pass

    def begin(self):
        pass

    def commit(self):
        pass

    def clear_transaction(self):
        pass

    @contextmanager
    def exception_handler(self, sql: str):
        try:
            yield
        except Exception as e:
            logger.error(f"Error running SQL: {sql}")
            logger.error(f"Exception: {e}")
            raise DbtRuntimeError(str(e))

