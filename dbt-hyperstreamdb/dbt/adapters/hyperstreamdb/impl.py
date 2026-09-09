# Portions of this code were inspired by fal-ai/dbt-datafusion
# Copyright (c) 2022 fal-ai
# Licensed under the MIT License (or corresponding open source license)

from typing import Optional, List, Any

from dbt.adapters.base.relation import BaseRelation, RelationType
from dbt.adapters.sql import SQLAdapter
from dbt.adapters.events.logging import AdapterLogger
import dbt.exceptions
import pandas as pd

from .connections import HyperStreamDBConnectionManager

logger = AdapterLogger("HyperStreamDB")

class HyperStreamDBAdapter(SQLAdapter):
    ConnectionManager = HyperStreamDBConnectionManager

    @classmethod
    def date_function(cls):
        return "datenow()"

    @classmethod
    def is_cancelable(cls) -> bool:
        return False

    def convert_boolean_type(self, column=None):
        return "boolean"

    def convert_date_type(self, column=None):
        return "date"

    def convert_datetime_type(self, column=None):
        raise dbt.exceptions.NotImplementedException(
            "`datetime` is not implemented for this adapter!"
        )

    def convert_number_type(self, column=None):
        return "double"

    def convert_text_type(self, column=None):
        return "string"

    def convert_time_type(self, column=None):
        return "time"

    def debug_query(self) -> None:
        self.execute("SELECT 1 as id")

    # Datafusion / Flight SQL currently prefers standard unquoted or double-quoted identifiers
    def quote(self, identifier: str) -> str:
        return f'"{identifier}"'

    def list_schemas(self, database: str) -> List[str]:
        # Using standard information schema to fetch schemas
        # DataFusion natively supports information_schema.schemata
        try:
            _, results = self.execute(
                f"SELECT schema_name FROM information_schema.schemata",
                fetch=True
            )
            return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Failed to list schemas: {e}")
            return ["public"]

    def list_relations_without_caching(
        self, schema_relation: BaseRelation
    ) -> List[BaseRelation]:
        schema = schema_relation.schema
        if not schema:
            schema = "public"
            
        sql = f"""
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = '{schema}'
        """
        try:
            _, results = self.execute(sql, fetch=True)
            relations = []
            for row in results:
                table_name = row[0]
                table_type = row[1]
                
                rel_type = RelationType.Table if "TABLE" in table_type else RelationType.View
                relations.append(self.Relation.create(
                    database=schema_relation.database,
                    schema=schema,
                    identifier=table_name,
                    type=rel_type
                ))
            return relations
        except Exception as e:
            logger.error(f"Failed to list relations: {e}")
            return []
