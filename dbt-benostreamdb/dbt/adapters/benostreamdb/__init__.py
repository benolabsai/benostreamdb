from dbt.adapters.base import AdapterPlugin
from dbt.include import benostreamdb

from .connections import BenoStreamDBConnectionManager
from .connections import BenoStreamDBCredentials
from .impl import BenoStreamDBAdapter

Plugin = AdapterPlugin(
    adapter=BenoStreamDBAdapter,
    credentials=BenoStreamDBCredentials,
    include_path=benostreamdb.PACKAGE_PATH,
)
