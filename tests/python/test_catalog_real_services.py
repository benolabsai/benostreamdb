# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

"""Catalog conformance against real catalog servers.

Every networked catalog client is exercised against the open-source server that
implements it:

| catalog | server |
|---|---|
| `rest` | open-source Unity Catalog Iceberg REST mount, and Apache Polaris (the engine behind Snowflake Open Catalog) |
| `unity` | open-source Unity Catalog (both Iceberg REST and the standard control plane) |
| `nessie` | Project Nessie (`ghcr.io/projectnessie/nessie`) |
| `hive` | Apache Hive Metastore standalone (`apache/hive`) |
| `jdbc` | PostgreSQL (`postgres`) |

`memory` needs no server and is covered in `test_catalog_bindings.py`.

Testing against the real implementation keeps the client honest: an embedded
mock can drift from the catalog's protocol and then pass while the client is
wrong.

`glue` is tested against the **AWS Emulator** (`aws/aws-emulator`), which
emulates the Glue Data Catalog API. Pointing a client at it needs three
test-only switches, all set from this file:

- `BSDB_SSRF_ALLOW_HOSTS=127.0.0.1` — the deliberate, loopback-only SSRF
  override (`src/core/storage.rs`); see `docs/CONFIGURATION.md`.
- `SSL_CERT_FILE=<emulator ca.pem>` — the emulator serves HTTPS with a
  self-signed CA it writes next to its credentials.
- `AWS_ENDPOINT_URL=https://127.0.0.1:<port>` plus the emulator's per-run keys.

`s3tables` remains excluded: both it and Glue's *Iceberg REST* data plane
(`https://glue.<region>.amazonaws.com/iceberg`) are not emulated. Measured
against `aws-emulator` LTS:

| request | result |
|---|---|
| `GET /iceberg/v1/config?warehouse=…` (Host: `glue.us-east-1.amazonaws.com`) | `400 SerializationException: missing required header x-amz-target` — routed to the Glue JSON-RPC surface, no Iceberg REST handler |
| `GET /iceberg/v1/config?warehouse=…` (Host: `s3tables.us-east-1.amazonaws.com`) | `501` — not implemented |

Our S3 Tables client only speaks that Iceberg REST mount, so there is nothing
for the emulator to serve. SigV4 request construction for both remains covered by
unit tests in `src/core/catalog/rest.rs`.

These tests need Docker and are skipped when it is unavailable. Override images
with `BSDB_UC_IMAGE`, `BSDB_NESSIE_IMAGE`, `BSDB_HIVE_IMAGE`, `BSDB_POSTGRES_IMAGE`.
"""

import json
import os
import platform
import shutil
import socket
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import pyarrow as pa
import pytest

import benostreamdb as bsdb

UC_IMAGE = os.environ.get("BSDB_UC_IMAGE", "unitycatalog/unitycatalog:latest")
NESSIE_IMAGE = os.environ.get(
    "BSDB_NESSIE_IMAGE", "ghcr.io/projectnessie/nessie:0.105.5"
)
HIVE_IMAGE = os.environ.get("BSDB_HIVE_IMAGE", "apache/hive:4.0.0")
POSTGRES_IMAGE = os.environ.get("BSDB_POSTGRES_IMAGE", "postgres:17")
POLARIS_IMAGE = os.environ.get("BSDB_POLARIS_IMAGE", "apache/polaris:latest")
POLARIS_CLIENT_ID = "root"
POLARIS_CLIENT_SECRET = "s3cr3t"
POLARIS_SCOPE = "PRINCIPAL_ROLE:ALL"

AEMU_RELEASE = os.environ.get("BSDB_AWS_EMULATOR_RELEASE", "latest")
AEMU_CACHE = Path(
    os.environ.get(
        "BSDB_AWS_EMULATOR_CACHE", Path(tempfile.gettempdir()) / "bsdb-aws-emulator"
    )
)


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


pytestmark = [
    pytest.mark.skipif(
        not _docker_available(), reason="docker is required for real catalog conformance"
    ),
    # A cold image pull happens inside a module-scoped fixture, which counts
    # against the first test's budget.
    pytest.mark.timeout(900),
]


# --------------------------------------------------------------------------- #
# Docker helpers
# --------------------------------------------------------------------------- #


def _docker(*args: str, timeout: int = 300) -> str:
    proc = subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True, timeout=timeout
    )
    return proc.stdout


def _start(name: str, image: str, container_port: int, *args: str) -> int:
    """Run `image` detached with a free host port; return the host port."""
    _docker(
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"127.0.0.1::{container_port}",
        *args,
        image,
    )
    published = _docker("port", name, str(container_port)).strip()
    return int(published.rsplit(":", 1)[1])


def _stop(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=120)


def _wait_tcp(port: int, timeout_s: int = 240) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(1)
    raise RuntimeError(f"port {port} never opened")


def _wait_http(url: str, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    last: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 - localhost
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last = exc
        time.sleep(1)
    raise RuntimeError(f"{url} never became ready: {last}")


def _post_json(url: str, payload: dict, token: str | None = None) -> None:
    body = json.dumps(payload).encode()
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30):  # noqa: S310 - localhost only
        pass


def _polaris_token(base_url: str) -> str:
    body = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "scope": POLARIS_SCOPE,
            "client_id": POLARIS_CLIENT_ID,
            "client_secret": POLARIS_CLIENT_SECRET,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/api/catalog/v1/oauth/tokens",
        data=body,
        headers={"content-type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - localhost
        return json.load(resp)["access_token"]


def _name(prefix: str) -> str:
    return f"bsdb-{prefix}-{uuid.uuid4().hex[:8]}"


def _wait_polaris_token(base_url: str, timeout_s: int = 240) -> str:
    """Poll the OAuth2 endpoint; a token means Polaris is accepting requests.

    Polaris does not expose a stable health path across versions, but the token
    endpoint is what every later call needs anyway.
    """
    deadline = time.time() + timeout_s
    last: Exception | None = None
    while time.time() < deadline:
        try:
            return _polaris_token(base_url)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            last = exc
        time.sleep(2)
    raise RuntimeError(f"polaris never issued a token: {last}")


def _aws_emulator_binary() -> Path:
    """Return the emulator binary, unpacking the LTS `.deb` without root if needed."""
    if platform.machine() not in ("x86_64", "aarch64"):
        pytest.skip(f"aws-emulator has no package for {platform.machine()}")
    if shutil.which("dpkg-deb") is None:
        pytest.skip("dpkg-deb is required to unpack the aws-emulator package")

    binary = AEMU_CACHE / "opt" / "aws-emulator" / "bin" / "aws-emulator"
    if binary.exists():
        return binary

    arch = "amd64" if platform.machine() == "x86_64" else "arm64"
    url = (
        "https://downloads.aws.com/aws-emulator/packages/aws-emulator"
        f"/LTS/{AEMU_RELEASE}/aws-emulator_{arch}.deb"
    )
    AEMU_CACHE.mkdir(parents=True, exist_ok=True)
    deb = AEMU_CACHE / "aws-emulator.deb"
    try:
        urllib.request.urlretrieve(url, deb)  # noqa: S310 - fixed AWS downloads host
        subprocess.run(
            ["dpkg-deb", "-x", str(deb), str(AEMU_CACHE)], check=True, timeout=600
        )
    except (urllib.error.URLError, OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"aws-emulator is not installable here: {exc}")
    finally:
        deb.unlink(missing_ok=True)
    return binary


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _parse_aws_credentials(path: Path) -> tuple[str, str]:
    key = secret = ""
    for line in path.read_text().splitlines():
        name, _, value = line.partition("=")
        name = name.strip()
        if name == "aws_access_key_id":
            key = value.strip()
        elif name == "aws_secret_access_key":
            secret = value.strip()
    return key, secret


def _wait_file(path: Path, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return
        time.sleep(0.5)
    raise RuntimeError(f"{path} never appeared")


def _wait_https(url: str, ca: Path, timeout_s: int = 180) -> None:
    context = ssl.create_default_context(cafile=str(ca))
    deadline = time.time() + timeout_s
    last: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5, context=context) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last = exc
        time.sleep(1)
    raise RuntimeError(f"{url} never became ready: {last}")


def _wait_hive(url: str, timeout_s: int = 300) -> None:
    """Poll until the metastore answers a real Thrift request.

    The Thrift socket binds long before the metastore finishes initialising its
    Derby schema, so a connect check alone is not readiness.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            bsdb.create_catalog("hive", {"url": url}).list_tables("default")
            return
        except RuntimeError as exc:
            message = str(exc)
            if "Transport" not in message and "UnexpectedEof" not in message:
                raise
        time.sleep(2)
    raise RuntimeError("hive metastore never served a request")


def _wait_pg(container: str, timeout_s: int = 120) -> None:
    """Poll `pg_isready`; the listening socket exists before the DB accepts."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        proc = subprocess.run(
            ["docker", "exec", container, "pg_isready", "-U", "postgres"],
            capture_output=True,
            timeout=30,
        )
        if proc.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError("postgres never became ready")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def unity_catalog(tmp_path_factory):
    """Open-source Unity Catalog; yields `(base_url, storage_root)`."""
    storage = tmp_path_factory.mktemp("unity_catalog") / "ext"
    storage.mkdir()

    name = _name("uc")
    # The server runs as uid 100 inside the image and writes the first metadata
    # file itself; running it as root with umask 000 keeps the shared location
    # writable by both the server and this test process.
    _docker(
        "run",
        "-d",
        "--name",
        name,
        "-p",
        "127.0.0.1::8080",
        "--user",
        "0",
        "--entrypoint",
        "sh",
        "-e",
        f"JAVA_TOOL_OPTIONS=-Dserver.external-local-roots={storage}",
        "-v",
        f"{storage}:{storage}",
        UC_IMAGE,
        "-c",
        "umask 000; exec ./bin/start-uc-server",
    )
    try:
        port = int(_docker("port", name, "8080").strip().rsplit(":", 1)[1])
        base_url = f"http://127.0.0.1:{port}"
        _wait_http(f"{base_url}/api/2.1/unity-catalog/catalogs")
        # The image ships a catalog named `unity`; the clients under test use `main`.
        _post_json(f"{base_url}/api/2.1/unity-catalog/catalogs", {"name": "main"})
        yield base_url, storage
    finally:
        _stop(name)


@pytest.fixture(scope="module")
def nessie():
    name = _name("nessie")
    port = _start(name, NESSIE_IMAGE, 19120)
    try:
        base_url = f"http://127.0.0.1:{port}"
        _wait_http(f"{base_url}/api/v2/config")
        yield base_url
    finally:
        _stop(name)


@pytest.fixture(scope="module")
def hive_metastore():
    name = _name("hive")
    # Standalone metastore on Derby; the first boot initialises the schema, which
    # takes a while. The Thrift socket accepts connections long before the
    # metastore answers RPCs, so readiness is a successful `list_tables`.
    port = _start(name, HIVE_IMAGE, 9083, "-e", "SERVICE_NAME=metastore")
    url = f"thrift://127.0.0.1:{port}"
    try:
        _wait_tcp(port, timeout_s=300)
        _wait_hive(url, timeout_s=300)
        yield url
    finally:
        _stop(name)


@pytest.fixture(scope="module")
def postgres():
    name = _name("pg")
    port = _start(
        name,
        POSTGRES_IMAGE,
        5432,
        "-e",
        "POSTGRES_PASSWORD=bsdb",
        "-e",
        "POSTGRES_DB=catalog",
    )
    try:
        _wait_tcp(port)
        _wait_pg(name, timeout_s=120)
        yield f"postgres://postgres:bsdb@127.0.0.1:{port}/catalog"
    finally:
        _stop(name)


@pytest.fixture(scope="module")
def polaris(tmp_path_factory):
    """Apache Polaris — the open-source engine behind Snowflake Open Catalog.

    Yields `(base_url, storage_root)`. The insecure-storage flags are the
    documented quickstart switches; without them the server refuses to start and
    rejects `FILE` catalogs.
    """
    storage = tmp_path_factory.mktemp("polaris") / "ext"
    storage.mkdir()

    name = _name("polaris")
    _docker(
        "run",
        "-d",
        "--name",
        name,
        "-p",
        "127.0.0.1::8181",
        "-e",
        f"POLARIS_BOOTSTRAP_CREDENTIALS=POLARIS,{POLARIS_CLIENT_ID},{POLARIS_CLIENT_SECRET}",
        "-e",
        'JAVA_OPTS_APPEND=-Dpolaris.readiness.ignore-severe-issues=true '
        '-Dpolaris.features."ALLOW_INSECURE_STORAGE_TYPES"=true '
        '-Dpolaris.features."SUPPORTED_CATALOG_STORAGE_TYPES"=["FILE","S3","GCS","AZURE"]',
        "-v",
        f"{storage}:{storage}",
        POLARIS_IMAGE,
    )
    try:
        port = int(_docker("port", name, "8181").strip().rsplit(":", 1)[1])
        base_url = f"http://127.0.0.1:{port}"
        token = _wait_polaris_token(base_url)
        _post_json(
            f"{base_url}/api/management/v1/catalogs",
            {
                "catalog": {
                    "name": "bsdb",
                    "type": "INTERNAL",
                    "readOnly": False,
                    "properties": {"default-base-location": f"file://{storage}"},
                    "storageConfigInfo": {
                        "storageType": "FILE",
                        "allowedLocations": [f"file://{storage}"],
                    },
                }
            },
            token,
        )
        yield base_url, storage
    finally:
        _stop(name)


@pytest.fixture(scope="module")
def aws_emulator(tmp_path_factory):
    """AWS Emulator serving the Glue Data Catalog API.

    Yields `(port, access_key, secret_key, ca_bundle)`. The emulator mints fresh
    credentials and a self-signed CA per run under `$HOME/.aws/emulator/<port>/`,
    so `HOME` is redirected into the test's temp tree.
    """
    binary = _aws_emulator_binary()
    home = tmp_path_factory.mktemp("aws-emulator")
    port = _free_port()
    env = dict(os.environ, HOME=str(home), EMULATOR_TELEMETRY="off")
    proc = subprocess.Popen(
        [str(binary), "-p", str(port), "-t", "off"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    state = home / ".aws" / "emulator" / str(port)
    try:
        _wait_file(state / "emulator_credentials")
        _wait_file(state / "ca.pem")
        _wait_https(f"https://127.0.0.1:{port}/__admin/health", state / "ca.pem")
        key, secret = _parse_aws_credentials(state / "emulator_credentials")
        yield port, key, secret, state / "ca.pem"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


# --------------------------------------------------------------------------- #
# Shared round-trip
# --------------------------------------------------------------------------- #


def _schema() -> bsdb.Schema:
    return bsdb.Schema.from_arrow(
        pa.schema(
            [
                pa.field("id", pa.int32(), nullable=False),
                pa.field("label", pa.string(), nullable=False),
            ]
        )
    )


def _roundtrip(cat, namespace: str, location) -> None:
    location.mkdir(mode=0o777, parents=True, exist_ok=True)
    (location / "metadata").mkdir(mode=0o777, exist_ok=True)
    location.chmod(0o777)
    (location / "metadata").chmod(0o777)

    cat.create_namespace(namespace)
    assert cat.namespace_exists(namespace) is True
    assert cat.list_tables(namespace) == []

    cat.create_table(namespace, "events", _schema(), f"file://{location}")
    assert cat.table_exists(namespace, "events") is True
    assert cat.list_tables(namespace) == ["events"]

    tbl = cat.load_table(namespace, "events")
    tbl.write(pa.table({"id": [1, 2, 3], "label": ["a", "b", "c"]}))
    tbl.commit()

    reloaded = cat.load_table(namespace, "events")
    assert reloaded.sql("SELECT COUNT(*) AS cnt FROM t")["cnt"].to_pylist() == [3]

    cat.drop_table(namespace, "events")
    assert cat.table_exists(namespace, "events") is False
    cat.drop_namespace(namespace)
    assert cat.namespace_exists(namespace) is False


# --------------------------------------------------------------------------- #
# Iceberg REST / Unity
# --------------------------------------------------------------------------- #


def test_iceberg_rest_catalog_roundtrip(unity_catalog):
    """Generic Iceberg REST client against the server's Iceberg REST mount."""
    base_url, storage = unity_catalog
    cat = bsdb.create_catalog(
        "rest",
        {"url": f"{base_url}/api/2.1/unity-catalog/iceberg", "warehouse": "main"},
    )
    _roundtrip(cat, "rest_ns", storage / "rest_tbl")


def test_unity_catalog_iceberg_rest_autodetect(unity_catalog):
    """Unity client finds `/iceberg` rather than the Databricks `/iceberg-rest`."""
    base_url, storage = unity_catalog
    cat = bsdb.create_catalog(
        "unity",
        {"url": base_url, "token": "unused", "warehouse": "main", "mode": "iceberg-rest"},
    )
    _roundtrip(cat, "main.unity_ns", storage / "unity_tbl")


def test_unity_catalog_explicit_mount_point(unity_catalog):
    """An explicit `.../iceberg` base URL is probed first and kept."""
    base_url, storage = unity_catalog
    cat = bsdb.create_catalog(
        "unity",
        {
            "url": f"{base_url}/api/2.1/unity-catalog/iceberg",
            "token": "unused",
            "warehouse": "main",
        },
    )
    _roundtrip(cat, "main.mount_ns", storage / "mount_tbl")


# --------------------------------------------------------------------------- #
# Nessie / Hive / JDBC
# --------------------------------------------------------------------------- #


def test_nessie_catalog_roundtrip(nessie, tmp_path):
    cat = bsdb.create_catalog("nessie", {"url": nessie})
    _roundtrip(cat, "prod", tmp_path / "nessie_tbl")


def test_hive_metastore_roundtrip(hive_metastore, tmp_path):
    cat = bsdb.create_catalog("hive", {"url": hive_metastore})
    _roundtrip(cat, "bsdb_prod", tmp_path / "hive_tbl")


def test_jdbc_postgres_roundtrip(postgres, tmp_path):
    cat = bsdb.create_catalog(
        "jdbc",
        {
            "uri": postgres,
            "warehouse": f"file://{tmp_path}/wh",
            "catalog_name": "bsdb",
        },
    )
    _roundtrip(cat, "prod", tmp_path / "pg_tbl")


def test_polaris_catalog_roundtrip(polaris):
    """Snowflake Open Catalog path: OAuth2 + `/v1/config` prefix discovery.

    Polaris only accepts table locations under the namespace directory, so the
    location is nested inside `{root}/prod`.
    """
    base_url, storage = polaris
    cat = bsdb.create_catalog(
        "rest",
        {
            "url": f"{base_url}/api/catalog",
            "warehouse": "bsdb",
            "credential": f"{POLARIS_CLIENT_ID}:{POLARIS_CLIENT_SECRET}",
            "scope": POLARIS_SCOPE,
        },
    )
    _roundtrip(cat, "prod", storage / "prod" / "tbl")


def test_glue_data_catalog_roundtrip(aws_emulator, tmp_path, monkeypatch):
    """Glue Data Catalog through the AWS SDK, against the AWS Emulator.

    Needs the loopback-only SSRF override (`BSDB_SSRF_ALLOW_HOSTS`) because the
    emulator listens on 127.0.0.1.
    """
    port, key, secret, ca = aws_emulator
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", key)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL", f"https://127.0.0.1:{port}")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    monkeypatch.setenv("BSDB_SSRF_ALLOW_HOSTS", "127.0.0.1")

    cat = bsdb.create_catalog("glue", {"region": "us-east-1"})
    _roundtrip(cat, "bsdb_prod", tmp_path / "glue_tbl")
