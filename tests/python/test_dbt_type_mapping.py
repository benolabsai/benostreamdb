"""Type mapping contract for the embedded dbt adapter.

dbt asks the adapter to translate warehouse types into SQL types for model
materializations. `datetime` used to raise `NotImplementedException`, which made
any model selecting a timestamp column fail at materialization time; it now maps
to `timestamp`, matching what the Trino connector exposes for the same columns.

Skipped when dbt is not installed (the ecosystem CI job installs it).
"""

import pytest

pytest.importorskip("dbt")

from dbt.adapters.benostreamdb.impl import BenoStreamDBAdapter  # noqa: E402


@pytest.fixture(scope="module")
def adapter():
    # The type converters do not touch instance state, so a bare instance keeps
    # the test free of connection/profile setup.
    return BenoStreamDBAdapter.__new__(BenoStreamDBAdapter)


def test_datetime_maps_to_timestamp(adapter):
    assert adapter.convert_datetime_type() == "timestamp"


def test_scalar_type_mappings_are_stable(adapter):
    assert adapter.convert_text_type() == "varchar"
    assert adapter.convert_number_type() == "double"
    assert adapter.convert_boolean_type() == "boolean"
    assert adapter.convert_date_type() == "date"
    assert adapter.convert_time_type() == "time"
