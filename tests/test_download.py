"""D1 tests: the registry is complete, each source is read right, and downloads resume."""

import hashlib
import json

import pytest
import requests

from catena.data import download as D

DATA = bytes(range(256)) * 4096  # 1 MiB of known bytes


# --------------------------------------------------------------------------- registry


def test_registry_has_every_dataset_we_plan_to_use():
    reg = D.registry()
    planned = {
        "cloudsen12plus",
        "kappaset",
        "l8_biome",
        "l7_irish",
        "cloudperu2",
        "iride_heo",
        "iride_eaglet2",
        "gf2ms_whu",
        "phisat2",
    }
    assert planned <= set(reg)


def test_every_dataset_names_a_sensor_spec():
    S = pytest.importorskip("catena.sensors.spec", reason="the sensor specs arrive in A2")
    sensors = {e["sensor"] for e in D.registry().values()}
    assert sensors - set(S.available()) <= {"gaofen2", "landsat7_etm"}  # specs to write


def test_registry_rejects_entries_it_cannot_use(tmp_path):
    bad = tmp_path / "sources.yaml"
    bad.write_text("x: {name: X, sensor: s, method: ftp}\n")
    with pytest.raises(ValueError, match="method"):
        D.registry(bad)
    bad.write_text("x: {name: X, sensor: s, method: zenodo}\n")
    with pytest.raises(ValueError, match="record"):
        D.registry(bad)


# ---------------------------------------------------------------- reading the sources


def test_zenodo_files_come_from_the_api_record():
    record = {
        "id": 7100327,
        "files": [
            {
                "key": "set.zip",
                "size": 10,
                "checksum": "md5:abc",
                "links": {"self": "https://zenodo.org/api/records/7100327/files/set.zip/content"},
            },
            {"key": "read me.txt", "size": 5, "checksum": "md5:def", "links": {}},
        ],
    }
    zipped, readme = D.zenodo_files(record)
    assert zipped.url.endswith("/files/set.zip/content")
    assert (zipped.size, zipped.md5) == (10, "abc")
    assert readme.url == "https://zenodo.org/records/7100327/files/read%20me.txt?download=1"


def test_links_on_a_page_are_found_and_made_absolute():
    page = (
        '<a href="/data/LC8_A.tar.gz">A</a> '
        "<a href='https://x.org/LC8_B.tar.gz?k=1&amp;t=2'>B</a> "
        '<a href="guide.pdf">guide</a> <a href="/data/LC8_A.tar.gz">A again</a>'
    )
    files = D.links_in(page, "https://www.usgs.gov/landsat/validation", r"\.tar\.gz$")
    assert [f.name for f in files] == ["LC8_A.tar.gz", "LC8_B.tar.gz"]
    assert files[0].url == "https://www.usgs.gov/data/LC8_A.tar.gz"
    assert files[1].url == "https://x.org/LC8_B.tar.gz?k=1&t=2"


# ------------------------------------------------------------------------ downloading


class FakeResponse:
    def __init__(self, body: bytes, status: int):
        self.status_code, self.body = status, body
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(self.status_code)

    def iter_content(self, size):
        for i in range(0, len(self.body), size):
            yield self.body[i : i + size]


class FakeServer:
    """Serves one file the way Zenodo does; can be told to ignore Range requests."""

    def __init__(self, data: bytes, honours_ranges: bool = True):
        self.data, self.honours_ranges, self.ranges = data, honours_ranges, []

    def get(self, url, headers=None, stream=False, timeout=None):
        wanted = (headers or {}).get("Range")
        self.ranges.append(wanted)
        if wanted and self.honours_ranges:
            start = int(wanted.removeprefix("bytes=").removesuffix("-"))
            return FakeResponse(self.data[start:], 206)
        return FakeResponse(self.data, 200)


def remote(md5: str | None = None) -> D.Remote:
    return D.Remote(
        "x.bin", "https://example.org/x.bin", len(DATA), md5 or hashlib.md5(DATA).hexdigest()
    )


@pytest.mark.parametrize("honours_ranges", [True, False])
def test_an_interrupted_download_finishes_whole(tmp_path, honours_ranges):
    (tmp_path / "x.bin.part").write_bytes(DATA[:300_000])  # left by an interruption
    server = FakeServer(DATA, honours_ranges)
    D.fetch(server, remote(), tmp_path / "x.bin")
    assert (tmp_path / "x.bin").read_bytes() == DATA
    assert server.ranges == ["bytes=300000-"]  # it asked only for the missing bytes
    assert not (tmp_path / "x.bin.part").exists()


def test_a_corrupt_download_is_thrown_away(tmp_path):
    with pytest.raises(OSError, match="md5"):
        D.fetch(FakeServer(DATA), remote(md5="0" * 32), tmp_path / "x.bin", retries=1)
    assert not (tmp_path / "x.bin").exists()
    assert not (tmp_path / "x.bin.part").exists()


def test_the_manifest_records_an_md5_even_when_the_source_gives_none(tmp_path):
    (tmp_path / "x.bin").write_bytes(DATA)
    entry = {"name": "X", "sensor": "s", "method": "page"}
    files = [D.Remote("x.bin", "https://example.org/x.bin")]  # a page gives no md5
    path = D.write_manifest(tmp_path, "x", entry, {"page": "https://example.org"}, files)
    listed = json.loads(path.read_text())["files"]
    assert listed == [
        {
            "name": "x.bin",
            "bytes": len(DATA),
            "md5": hashlib.md5(DATA).hexdigest(),
            "url": "https://example.org/x.bin",
        }
    ]
