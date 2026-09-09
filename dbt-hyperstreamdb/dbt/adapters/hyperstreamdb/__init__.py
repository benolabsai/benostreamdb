from dbt.adapters.base import AdapterPlugin
from dbt.include import hyperstreamdb

from .connections import HyperStreamDBConnectionManager
from .connections import HyperStreamDBCredentials
from .impl import HyperStreamDBAdapter

Plugin = AdapterPlugin(
    adapter=HyperStreamDBAdapter,
    credentials=HyperStreamDBCredentials,
    include_path=hyperstreamdb.PACKAGE_PATH,
)
