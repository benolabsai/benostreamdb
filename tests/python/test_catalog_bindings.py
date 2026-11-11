# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

"""Python catalog bindings conformance tests for memory, jdbc, rest, s3tables, and unity."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.parse import parse_qs, unquote, urlparse

import pyarrow as pa
import pytest

import benostreamdb as bsdb


def _schema() -> bsdb.Schema:
    return bsdb.Schema.from_arrow(
        pa.schema(
            [
                pa.field("id", pa.int32(), nullable=False),
                pa.field("label", pa.string(), nullable=False),
            ]
        )
    )


def _start_mock_server():
    """Start an embedded HTTP mock supporting Iceberg REST (OAuth2/SigV4/config) & Unity REST."""
    state = {
        "lock": threading.Lock(),
        "tables": {},  # (prefix, ns, name) -> {"location": str, "metadata_location": str, "metadata": dict}
        "namespaces": set(),  # (prefix, ns)
        "unity_tables": {},  # full_name -> dict
        "unity_schemas": set(),  # full_name
        "token_calls": 0,
        "config_calls": 0,
        "sigv4_headers": [],
        "unity_patches": 0,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            return

        def _send_json(self, code: int, payload: dict):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8")) if raw else {}

        def _record_sigv4(self):
            auth = self.headers.get("Authorization", "")
            if auth.startswith("AWS4-HMAC-SHA256"):
                with state["lock"]:
                    state["sigv4_headers"].append(
                        {
                            "auth": auth,
                            "date": self.headers.get("x-amz-date", ""),
                            "sha256": self.headers.get("x-amz-content-sha256", ""),
                        }
                    )

        def do_HEAD(self):  # noqa: N802
            parsed = urlparse(self.path)
            parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
            # /v1/<prefix>/namespaces/<ns>
            if len(parts) == 4 and parts[0] == "v1" and parts[2] == "namespaces":
                with state["lock"]:
                    exists = (parts[1], parts[3]) in state["namespaces"]
                self.send_response(200 if exists else 404)
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def do_GET(self):  # noqa: N802
            self._record_sigv4()
            parsed = urlparse(self.path)
            parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]

            # Iceberg /v1/config?warehouse=...
            if parts == ["v1", "config"]:
                qs = parse_qs(parsed.query)
                wh = qs.get("warehouse", [""])[0]
                prefix = "s3tables_pfx" if wh.startswith("arn:aws:s3tables:") else "unity_pfx"
                with state["lock"]:
                    state["config_calls"] += 1
                self._send_json(200, {"overrides": {"prefix": prefix}, "defaults": {"clients": "2"}})
                return

            # Unity standard schema check: /api/2.1/unity-catalog/schemas/<full>
            if parts[:3] == ["api", "2.1", "unity-catalog"] and len(parts) == 5 and parts[3] == "schemas":
                with state["lock"]:
                    exists = parts[4] in state["unity_schemas"]
                if exists:
                    self._send_json(200, {"full_name": parts[4]})
                else:
                    self._send_json(404, {"message": "schema not found"})
                return

            # Unity standard list tables: /api/2.1/unity-catalog/tables?catalog_name=...&schema_name=...
            if parts == ["api", "2.1", "unity-catalog", "tables"]:
                qs = parse_qs(parsed.query)
                cat = qs.get("catalog_name", ["main"])[0]
                sch = qs.get("schema_name", ["default"])[0]
                prefix_dot = f"{cat}.{sch}."
                with state["lock"]:
                    tables = [
                        {"name": v["name"], "catalog_name": cat, "schema_name": sch,
                         "table_type": "EXTERNAL", "data_source_format": "ICEBERG",
                         "storage_location": v["storage_location"], "properties": v["properties"]}
                        for k, v in state["unity_tables"].items()
                        if k.startswith(prefix_dot)
                    ]
                self._send_json(200, {"tables": tables})
                return

            # Unity standard load table: /api/2.1/unity-catalog/tables/<full>
            if parts[:3] == ["api", "2.1", "unity-catalog"] and len(parts) == 5 and parts[3] == "tables":
                with state["lock"]:
                    tbl = state["unity_tables"].get(parts[4])
                if tbl is None:
                    self._send_json(404, {"message": "not found"})
                else:
                    self._send_json(200, tbl)
                return

            # Iceberg REST list tables: /v1/<prefix>/namespaces/<ns>/tables
            if len(parts) == 5 and parts[0] == "v1" and parts[2] == "namespaces" and parts[4] == "tables":
                pfx, ns = parts[1], parts[3]
                with state["lock"]:
                    names = sorted(t for (p, n, t) in state["tables"] if p == pfx and n == ns)
                self._send_json(200, {"identifiers": [{"namespace": [ns], "name": n} for n in names]})
                return

            # Iceberg REST load table: /v1/<prefix>/namespaces/<ns>/tables/<tbl>
            if len(parts) == 6 and parts[0] == "v1" and parts[2] == "namespaces" and parts[4] == "tables":
                pfx, ns, tbl = parts[1], parts[3], parts[5]
                with state["lock"]:
                    rec = state["tables"].get((pfx, ns, tbl))
                if rec is None:
                    self._send_json(404, {"error": {"message": "not found", "code": 404}})
                else:
                    self._send_json(
                        200,
                        {"metadata-location": rec["metadata_location"], "metadata": rec["metadata"]},
                    )
                return

            self._send_json(404, {"message": f"unhandled GET {parsed.path}"})

        def do_POST(self):  # noqa: N802
            self._record_sigv4()
            parsed = urlparse(self.path)
            parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]

            if parts == ["v1", "oauth", "tokens"]:
                with state["lock"]:
                    state["token_calls"] += 1
                self._send_json(200, {"access_token": "py-mock-token", "token_type": "Bearer", "expires_in": 3600})
                return

            body = self._read_json()

            # Unity standard schema create
            if parts == ["api", "2.1", "unity-catalog", "schemas"]:
                full = f"{body['catalog_name']}.{body['name']}"
                with state["lock"]:
                    state["unity_schemas"].add(full)
                self._send_json(200, {"full_name": full})
                return

            # Unity standard table create
            if parts == ["api", "2.1", "unity-catalog", "tables"]:
                full = f"{body['catalog_name']}.{body['schema_name']}.{body['name']}"
                with state["lock"]:
                    state["unity_tables"][full] = body
                self._send_json(200, body)
                return

            # Iceberg REST create namespace: /v1/<prefix>/namespaces
            if len(parts) == 3 and parts[0] == "v1" and parts[2] == "namespaces":
                ns = ".".join(body["namespace"])
                with state["lock"]:
                    state["namespaces"].add((parts[1], ns))
                self._send_json(200, {"namespace": body["namespace"]})
                return

            # Iceberg REST create table: /v1/<prefix>/namespaces/<ns>/tables
            if len(parts) == 5 and parts[0] == "v1" and parts[2] == "namespaces" and parts[4] == "tables":
                pfx, ns = parts[1], parts[3]
                name = body["name"]
                loc = body["location"]
                # Read metadata from disk initialized by BenoStream / Table::create
                t = bsdb.Table.create(loc, _schema()) if False else None  # noqa: F841
                meta_loc = f"{loc.rstrip('/')}/metadata/v1.metadata.json"
                # Ensure minimal metadata exists so REST mock returns valid metadata JSON
                import pathlib
                p = pathlib.Path(loc.removeprefix("file://"))
                meta_files = sorted((p / "metadata").glob("v*.metadata.json"))
                if not meta_files:
                    bsdb.Table.create(
                        loc,
                        bsdb.Schema.from_arrow(
                            pa.schema([pa.field("id", pa.int32(), False), pa.field("label", pa.string(), False)])
                        ),
                    )
                    meta_files = sorted((p / "metadata").glob("v*.metadata.json"))
                meta_path = meta_files[-1]
                meta_loc = f"file://{meta_path}"
                meta_json = json.loads(meta_path.read_text())
                with state["lock"]:
                    if (pfx, ns, name) in state["tables"]:
                        self._send_json(409, {"error": {"message": "already exists", "code": 409}})
                        return
                    state["tables"][(pfx, ns, name)] = {
                        "location": loc,
                        "metadata_location": meta_loc,
                        "metadata": meta_json,
                    }
                self._send_json(200, {"metadata-location": meta_loc, "metadata": meta_json})
                return

            # Iceberg REST commit table: /v1/<prefix>/namespaces/<ns>/tables/<tbl>
            if len(parts) == 6 and parts[0] == "v1" and parts[2] == "namespaces" and parts[4] == "tables":
                pfx, ns, tbl = parts[1], parts[3], parts[5]
                with state["lock"]:
                    rec = state["tables"].get((pfx, ns, tbl))
                    if rec is None:
                        self._send_json(404, {"error": {"message": "not found", "code": 404}})
                        return
                    for req in body.get("requirements", []):
                        if req.get("type") == "assert-ref-snapshot-id":
                            if req.get("snapshot-id") != rec["metadata"].get("current-snapshot-id"):
                                self._send_json(409, {"error": {"message": "CAS conflict", "code": 409}})
                                return
                    for upd in body.get("updates", []):
                        if upd.get("action") == "add-snapshot":
                            snap = upd["snapshot"]
                            rec["metadata"].setdefault("snapshots", []).append(snap)
                            rec["metadata"]["current-snapshot-id"] = snap["snapshot-id"]
                        elif upd.get("action") == "set-current-snapshot":
                            rec["metadata"]["current-snapshot-id"] = upd["snapshot-id"]
                        elif upd.get("action") == "set-metadata-location":
                            rec["metadata_location"] = upd["metadata-location"]
                self._send_json(200, {"metadata-location": rec["metadata_location"], "metadata": rec["metadata"]})
                return

            self._send_json(404, {"message": f"unhandled POST {parsed.path}"})

        def do_PATCH(self):  # noqa: N802
            parsed = urlparse(self.path)
            parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
            body = self._read_json()
            if parts[:3] == ["api", "2.1", "unity-catalog"] and len(parts) == 5 and parts[3] == "tables":
                with state["lock"]:
                    tbl = state["unity_tables"].get(parts[4])
                    if tbl is None:
                        self._send_json(404, {"message": "not found"})
                        return
                    tbl["properties"] = body.get("properties", tbl.get("properties", {}))
                    state["unity_patches"] += 1
                self._send_json(200, tbl)
                return
            self._send_json(404, {"message": f"unhandled PATCH {parsed.path}"})

        def do_DELETE(self):  # noqa: N802
            self._record_sigv4()
            parsed = urlparse(self.path)
            parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
            if len(parts) == 4 and parts[0] == "v1" and parts[2] == "namespaces":
                with state["lock"]:
                    has_tbls = any(p == parts[1] and n == parts[3] for (p, n, _) in state["tables"])
                    if has_tbls:
                        self._send_json(409, {"error": {"message": "not empty", "code": 409}})
                        return
                    state["namespaces"].discard((parts[1], parts[3]))
                self.send_response(204)
                self.end_headers()
                return
            if len(parts) == 6 and parts[0] == "v1" and parts[2] == "namespaces" and parts[4] == "tables":
                with state["lock"]:
                    state["tables"].pop((parts[1], parts[3], parts[5]), None)
                self.send_response(204)
                self.end_headers()
                return
            if parts[:3] == ["api", "2.1", "unity-catalog"] and len(parts) == 5 and parts[3] == "tables":
                with state["lock"]:
                    state["unity_tables"].pop(parts[4], None)
                self.send_response(204)
                self.end_headers()
                return
            if parts[:3] == ["api", "2.1", "unity-catalog"] and len(parts) == 5 and parts[3] == "schemas":
                with state["lock"]:
                    state["unity_schemas"].discard(parts[4])
                self.send_response(204)
                self.end_headers()
                return
            self._send_json(404, {"message": f"unhandled DELETE {parsed.path}"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, state


def test_memory_catalog_python_roundtrip(tmp_path):
    warehouse = f"file://{tmp_path}/wh"
    cat = bsdb.create_catalog(
        "memory", {"uri": f"memory://py_test_{tmp_path.name}", "warehouse": warehouse}
    )
    assert isinstance(cat, bsdb.PyMemoryCatalog)

    cat.create_namespace("analytics")
    assert cat.namespace_exists("analytics") is True
    assert cat.list_tables("analytics") == []

    cat.create_table("analytics", "events", _schema())
    assert cat.table_exists("analytics", "events") is True
    assert cat.list_tables("analytics") == ["events"]

    tbl = cat.load_table("analytics", "events")
    tbl.write(pa.table({"id": [1, 2, 3], "label": ["a", "b", "c"]}))
    tbl.commit()

    tbl2 = cat.load_table("analytics", "events")
    res = tbl2.sql("SELECT COUNT(*) AS cnt FROM t")
    assert res["cnt"].to_pylist() == [3]

    with pytest.raises(RuntimeError):
        cat.drop_namespace("analytics")

    cat.drop_table("analytics", "events")
    assert cat.table_exists("analytics", "events") is False
    cat.drop_namespace("analytics")
    assert cat.namespace_exists("analytics") is False


def test_jdbc_sqlite_catalog_python_roundtrip(tmp_path):
    warehouse = f"file://{tmp_path}/wh"
    cat = bsdb.create_catalog(
        "jdbc",
        {
            "uri": "sqlite::memory:",
            "warehouse": warehouse,
            "catalog_name": "py_jdbc",
        },
    )

    cat.create_namespace("prod")
    assert cat.namespace_exists("prod") is True

    cat.create_table("prod", "metrics", _schema())
    assert cat.table_exists("prod", "metrics") is True
    assert cat.list_tables("prod") == ["metrics"]

    tbl = cat.load_table("prod", "metrics")
    tbl.write(pa.table({"id": [10, 20], "label": ["x", "y"]}))
    tbl.commit()

    tbl2 = cat.load_table("prod", "metrics")
    res = tbl2.sql("SELECT SUM(id) AS total FROM t")
    assert res["total"].to_pylist() == [30]

    cat.drop_table("prod", "metrics")
    assert cat.table_exists("prod", "metrics") is False
    cat.drop_namespace("prod")
    assert cat.namespace_exists("prod") is False


def test_rest_oauth2_and_s3tables_sigv4_python_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

    server, state = _start_mock_server()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        # 1. PyRestCatalog with OAuth2 client credentials
        rest_cat = bsdb.create_catalog(
            "rest",
            {
                "url": base_url,
                "prefix": "polaris",
                "credential": "client_id:client_secret",
            },
        )
        assert isinstance(rest_cat, bsdb.PyRestCatalog)
        rest_cat.create_namespace("analytics")
        assert rest_cat.namespace_exists("analytics") is True
        rest_loc = f"file://{tmp_path}/rest_tbl"
        rest_cat.create_table("analytics", "events", _schema(), rest_loc)
        assert rest_cat.list_tables("analytics") == ["events"]

        tbl = rest_cat.load_table("analytics", "events")
        tbl.write(pa.table({"id": [1, 2], "label": ["r1", "r2"]}))
        tbl.commit()
        assert state["token_calls"] == 1
        assert rest_cat.load_table("analytics", "events").sql("SELECT COUNT(*) AS cnt FROM t")["cnt"].to_pylist() == [2]

        # 2. PyS3TablesCatalog with AWS SigV4 + /v1/config prefix discovery
        arn = "arn:aws:s3tables:us-east-1:123456789012:bucket/test-bucket"
        s3_cat = bsdb.create_catalog(
            "s3tables",
            {"warehouse": arn, "url": base_url},
        )
        assert isinstance(s3_cat, bsdb.PyS3TablesCatalog)
        s3_cat.create_namespace("lake")
        s3_loc = f"file://{tmp_path}/s3_tbl"
        s3_cat.create_table("lake", "items", _schema(), s3_loc)
        assert s3_cat.list_tables("lake") == ["items"]
        t_s3 = s3_cat.load_table("lake", "items")
        t_s3.write(pa.table({"id": [5, 6, 7], "label": ["a", "b", "c"]}))
        t_s3.commit()
        assert state["config_calls"] >= 1
        assert any("/us-east-1/s3tables/aws4_request" in h["auth"] and h["sha256"] for h in state["sigv4_headers"])

        s3_cat.drop_table("lake", "items")
        assert s3_cat.table_exists("lake", "items") is False
        s3_cat.drop_namespace("lake")
    finally:
        server.shutdown()


def test_unity_standard_and_iceberg_rest_python_roundtrip(tmp_path):
    server, state = _start_mock_server()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        # 1. Standard Unity Tables REST mode (verifies PyTable::load_from_catalog commits PATCH back)
        unity_std = bsdb.create_catalog(
            "unity",
            {"url": base_url, "token": "pat-token", "warehouse": "main"},
        )
        assert isinstance(unity_std, bsdb.PyUnityCatalog)
        unity_std.create_namespace("main.prod")
        assert unity_std.namespace_exists("main.prod") is True
        std_loc = f"file://{tmp_path}/unity_std"
        unity_std.create_table("main.prod", "orders", _schema(), std_loc)
        assert unity_std.list_tables("main.prod") == ["orders"]

        tbl = unity_std.load_table("main.prod", "orders")
        tbl.write(pa.table({"id": [11, 22], "label": ["u1", "u2"]}))
        tbl.commit()
        assert state["unity_patches"] >= 1
        assert "metadata_location" in state["unity_tables"]["main.prod.orders"]["properties"]
        unity_std.drop_table("main.prod", "orders")
        unity_std.drop_namespace("main.prod")

        # 2. Databricks Unity Iceberg REST mode
        unity_rest = bsdb.create_catalog(
            "unity",
            {
                "url": base_url,
                "token": "pat-token",
                "mode": "iceberg-rest",
                "warehouse": "main",
            },
        )
        unity_rest.create_namespace("main.analytics")
        rest_loc = f"file://{tmp_path}/unity_rest"
        unity_rest.create_table("main.analytics", "events", _schema(), rest_loc)
        assert unity_rest.list_tables("main.analytics") == ["events"]
        tbl_r = unity_rest.load_table("main.analytics", "events")
        tbl_r.write(pa.table({"id": [99], "label": ["ice"]}))
        tbl_r.commit()
        assert tbl_r.sql("SELECT COUNT(*) AS cnt FROM t")["cnt"].to_pylist() == [1]
    finally:
        server.shutdown()
