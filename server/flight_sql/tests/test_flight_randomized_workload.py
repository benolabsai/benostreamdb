"""Randomized differential workload over the Flight SQL server.

Mimics ``tests/test_randomized_differential_workload.rs`` but drives the
gateway over the wire: a deterministic LCG generates a sequence of
INSERT / DELETE / OPTIMIZE / VACUUM / index / primary-key operations, and after
every step the table's visible id set is compared against an independent
ground-truth model kept in Python.

Run with the server listening on ``grpc://localhost:50051``::

    BSDB_WAREHOUSE=/tmp/bsdb-warehouse ./target/debug/benostreamdb-flight &
    pytest -v server/flight_sql/tests/test_flight_randomized_workload.py

Environment:
    BSDB_WORKLOAD_STEPS  number of randomized steps (default 100)
    BSDB_WORKLOAD_SEED   LCG seed (default 12345)
"""

import os

import adbc_driver_flightsql.dbapi as flight_sql

STEPS = int(os.environ.get("BSDB_WORKLOAD_STEPS", "100"))
SEED = int(os.environ.get("BSDB_WORKLOAD_SEED", "12345"))

TABLE = "mydb.myschema.workload"


class Lcg:
    """A tiny deterministic LCG, matching the Rust test's generator."""

    def __init__(self, seed: int) -> None:
        self.state = (seed * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.state = (
            self.state * 6364136223846793005 + 1442695040888963407
        ) & 0xFFFFFFFFFFFFFFFF
        return self.state >> 33

    def below(self, n: int) -> int:
        return 0 if n == 0 else self.next() % n


def visible_ids(cur) -> set:
    cur.execute(f"SELECT id FROM {TABLE};")
    return {row[0] for row in cur.fetchall()}


def main() -> None:
    rng = Lcg(SEED)
    model: set = set()
    next_id = 0

    with flight_sql.connect(uri="grpc://localhost:50051") as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE DATABASE IF NOT EXISTS mydb;")
            cur.execute("CREATE SCHEMA IF NOT EXISTS mydb.myschema;")
            cur.execute(f"DROP TABLE IF EXISTS {TABLE};")
            cur.execute(f"CREATE TABLE {TABLE} (id INT, value DOUBLE);")

            for step in range(STEPS):
                op = rng.below(100)

                if op < 40:
                    # INSERT a small batch of fresh ids.
                    n = 1 + rng.below(5)
                    ids = list(range(next_id, next_id + n))
                    next_id += n
                    values = ", ".join(f"({i}, {float(i)})" for i in ids)
                    cur.execute(f"INSERT INTO {TABLE} VALUES {values};")
                    model.update(ids)

                elif op < 60 and model:
                    # DELETE a random id.
                    victim = sorted(model)[rng.below(len(model))]
                    cur.execute(f"DELETE FROM {TABLE} WHERE id = {victim};")
                    model.discard(victim)

                elif op < 70:
                    cur.execute(f"OPTIMIZE TABLE {TABLE};")

                elif op < 78:
                    cur.execute(f"VACUUM {TABLE};")

                elif op < 86:
                    cur.execute(f"CREATE INDEX ON {TABLE} (id);")

                elif op < 90:
                    cur.execute(f"ALTER TABLE {TABLE} DROP INDEX id;")

                elif op < 95:
                    cur.execute(f"ALTER TABLE {TABLE} ADD PRIMARY KEY (id);")

                else:
                    cur.execute(f"ALTER TABLE {TABLE} DROP PRIMARY KEY;")

                got = visible_ids(cur)
                assert got == model, (
                    f"step {step} (op {op}): divergence\n"
                    f"  table = {sorted(got)}\n"
                    f"  model = {sorted(model)}"
                )

            print(f"OK: {STEPS} randomized steps, no divergence (seed={SEED})")


if __name__ == "__main__":
    main()
