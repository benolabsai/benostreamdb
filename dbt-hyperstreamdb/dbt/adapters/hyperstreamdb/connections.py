# Portions of this code were inspired by fal-ai/dbt-datafusion
# Copyright (c) 2022 fal-ai
# Licensed under the MIT License (or corresponding open source license)

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import adbc_driver_flightsql.dbapi as flight_sql
from dbt.adapters.contracts.connection import Credentials, Connection, AdapterResponse
from dbt.adapters.sql import SQLConnectionManager
from dbt.exceptions import DbtRuntimeError
from dbt.adapters.events.logging import AdapterLogger

logger = AdapterLogger("HyperStreamDB")


@dataclass
class HyperStreamDBCredentials(Credentials):
    host: str = "localhost"
    port: int = 50051
    database: Optional[str] = None
    schema: str = "public"

    @property
    def type(self):
        return "hyperstreamdb"

    @property
    def unique_field(self):
        return self.host

    def _connection_keys(self):
        return ("host", "port", "database", "schema")


class HyperStreamDBConnectionManager(SQLConnectionManager):
    TYPE = "hyperstreamdb"

    @classmethod
    def open(cls, connection: Connection) -> Connection:
        if connection.state == "open":
            logger.debug("Connection is already open, skipping open.")
            return connection

        credentials = connection.credentials
        uri = f"grpc://{credentials.host}:{credentials.port}"

        try:
            handle = flight_sql.connect(uri=uri)
            connection.handle = handle
            connection.state = "open"
        except Exception as e:
            logger.error(f"Error connecting to HyperStreamDB at {uri}: {e}")
            connection.handle = None
            connection.state = "fail"
            raise DbtRuntimeError(f"Failed to connect to HyperStreamDB: {e}")

        return connection

    @classmethod
    def get_response(cls, cursor: Any) -> AdapterResponse:
        return AdapterResponse(_message="OK")

    def cancel(self, connection: Connection):
        # ADBC cursor might have a cancel method, or we just pass
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
