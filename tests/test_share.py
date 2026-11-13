"""Keeping the work: HEO's and Eaglet II's data only ever to a private repository, the rest
packed into the public archives (what goes in each, the same files making the same archive),
and a Zenodo draft made, never published."""

import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from catena import share as S

FILES = {
    "raw/iride_heo/IMH12.zip": b"heo",
    "raw/iride_eaglet2/IME03.zip": b"eaglet",
    "labels/iride_heo/IMH12/label.tif": b"l1",
    "labels/iride_eaglet2/IME03/label.tif": b"l2",
    "chips/iride_heo/IMH12/bands.tif": b"b1",
    "raw/phisat2/products/P1.tif": b"phi",
    "labels/phisat2/P1/label.tif": b"l3",
    "chips/l8_biome/LC8/bands.tif": b"public dataset",
    "runs/big/seed0/final.pt": b"weights",
    "runs/big/seed0/config.json": b"{}",
    "runs/big/seed0/last.pt": b"resume",
    "runs/big/seed0/train.log": b"log",
    "runs/big/seed0/metrics.jsonl": b"{}",
    "runs/big/maps/IMH12.png": b"a picture of HEO",
    "runs/arms/chain/seed1/final.pt": b"weights",
    "runs/arms/chain/seed1/config.json": b"{}",
    "runs/pilot/seed0/final.pt": b"old",
    "runs/compare/all.csv": b"test,model",
    "runs/light/light.csv": b"set,scene,lat,lon",
    "runs/light/s2_earlier.csv": b"camera,window",
    "raw/iride_heo/.cache/huggingface/x": b"progress",
}


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "catena_data"
    for name, content in FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    return root


def test_what_goes_where(data):
    private = [str(p) for p in S.files_under(data, S.PRIVATE)]
    assert private == sorted(
        [
            "chips/iride_heo/IMH12/bands.tif",
            "labels/iride_eaglet2/IME03/label.tif",
            "labels/iride_heo/IMH12/label.tif",
            "raw/iride_eaglet2/IME03.zip",
            "raw/iride_heo/IMH12.zip",
        ]
    )  # no cache
    archives = {k: [str(p) for p in v] for k, v in S.public_files(data).items()}
    assert list(archives) == list(S.ARCHIVES)
    assert archives["phisat2.tar"] == ["labels/phisat2/P1/label.tif", "raw/phisat2/products/P1.tif"]
    assert archives["models.tar"] == [
        "runs/arms/chain/seed1/config.json",
        "runs/arms/chain/seed1/final.pt",
        "runs/big/seed0/config.json",
        "runs/big/seed0/final.pt",
    ]  # the runs the paper uses; no last.pt
    results = set(archives["results.tar"])
    assert results == {
        "runs/arms/chain/seed1/config.json",
        "runs/big/seed0/config.json",
        "runs/big/seed0/metrics.jsonl",
        "runs/big/seed0/train.log",
        "runs/compare/all.csv",
        "runs/light/s2_earlier.csv",
    }  # no picture, no weights, no light.csv
    every = {p for v in archives.values() for p in v}
    assert not every & set(private) and not any("l8_biome" in p for p in every)
    S.check_apart(S.public_files(data))
    with pytest.raises(SystemExit):
        S.check_apart({"results.tar": [S.Path("raw/iride_eaglet2/IME03.zip")]})


def test_the_archives(data, tmp_path):
    archives = S.public_files(data)
    made = S.pack(data, archives, tmp_path / "stage")
    assert [p.name for p in made] == [*S.ARCHIVES, "README.md"]
    with tarfile.open(made[1]) as tar:
        assert tar.getnames() == [str(p) for p in archives["models.tar"]]
        assert tar.extractfile("runs/big/seed0/final.pt").read() == b"weights"
    before = [p.read_bytes() for p in made]
    again = S.pack(data, archives, tmp_path / "stage")
    assert [p.read_bytes() for p in again] == before  # the same files, the same archives
    text = made[-1].read_text()
    assert "models.tar" in text and "not here" in text


class FakeHub:
    def __init__(self, private=True):
        self.private, self.calls = private, []

    def create_repo(self, repo, **kwargs):
        self.calls.append(("create", repo, kwargs))

    def repo_info(self, repo, **kwargs):
        return type("Info", (), {"private": self.private})()

    def upload_large_folder(self, **kwargs):
        self.calls.append(("upload", kwargs["repo_id"], kwargs))


def test_private_data_only_to_a_private_repository(data):
    api = FakeHub(private=False)  # a repository someone made public
    with pytest.raises(SystemExit):
        S.upload_private(api, "me/catena-private", data, 2)
    assert not [c for c in api.calls if c[0] == "upload"]
    api = FakeHub()
    S.upload_private(api, "me/catena-private", data, 2)
    (create,) = [c for c in api.calls if c[0] == "create"][:1]
    assert create[2]["private"] is True and create[2]["repo_type"] == "dataset"
    (upload,) = [c for c in api.calls if c[0] == "upload"]
    assert upload[2]["allow_patterns"] == [f"{f}/*" for f in S.PRIVATE]
    assert upload[2]["folder_path"] == str(data) and upload[2]["repo_type"] == "dataset"


def test_a_dry_run_sends_nothing(data, monkeypatch, capsys):
    def no(*args, **kwargs):
        raise AssertionError("nothing may be sent")

    monkeypatch.setattr(S, "hub", no)
    monkeypatch.setattr(S, "zenodo_draft", no)
    S.main([str(data), "--private", "--public", "--zenodo", "--dry-run"])
    out = capsys.readouterr().out
    assert "private (HEO, Eaglet II): 5 files" in out and "models.tar" in out
    assert not (data.parent / "catena_share").exists()
    with pytest.raises(SystemExit):
        S.main([str(data)])  # nothing asked for


def test_a_zenodo_draft(data, tmp_path):
    got = {"auth": set(), "files": {}}

    class Zenodo(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, body, code=200):
            text = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(text)))
            self.end_headers()
            self.wfile.write(text)

        def do_POST(self):
            got["auth"].add(self.headers["Authorization"])
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            host = f"http://127.0.0.1:{self.server.server_port}"
            self.answer({"id": 7, "links": {"bucket": f"{host}/b/7", "html": f"{host}/d/7"}}, 201)

        def do_PUT(self):
            got["auth"].add(self.headers["Authorization"])
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got["files"][self.path] = body
            self.answer({"key": self.path.rsplit("/", 1)[-1]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Zenodo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        made = S.pack(data, S.public_files(data), tmp_path / "stage")
        page = S.zenodo_draft(made, "secret", host=f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
    assert page.endswith("/d/7") and got["auth"] == {"Bearer secret"}
    assert set(got["files"]) == {f"/b/7/{p.name}" for p in made}
    assert got["files"]["/b/7/README.md"] == made[-1].read_bytes()
