"""D1: download the cloud datasets, resume after interruptions, record what we got.

Every dataset has one entry in sources.yaml saying where it comes from. Nothing is
downloaded blind: `plan` lists the files and their sizes; `get` downloads them into
$CATENA_DATA/raw/<key>/ and writes MANIFEST.json there (source, version, sizes,
checksums, date), so the paper can say exactly which data it used.

    python -m catena.data.download list
    python -m catena.data.download plan l8_biome l7_irish
    python -m catena.data.download get l8_biome
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import html
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import requests
import yaml
from tqdm import tqdm

# the keys each method needs, besides name and sensor
METHODS = {
    "zenodo": {"record"},
    "hf": {"repo", "patterns"},
    "page": {"page", "pattern"},
    "manual": {"note"},
}
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) catena-downloader/0.1"}
CHUNK = 1 << 20  # read and write 1 MiB at a time
ZENODO = "https://zenodo.org"


@dataclass
class Remote:
    """One file to download. Size and md5 stay None when the source doesn't give them."""

    name: str
    url: str
    size: int | None = None
    md5: str | None = None


# ---------------------------------------------------------------------------- registry


def registry(path: str | Path | None = None) -> dict:
    """The datasets in sources.yaml, each checked for the keys its method needs."""
    if path is None:
        text = resources.files("catena.data").joinpath("sources.yaml").read_text("utf-8")
    else:
        text = Path(path).read_text("utf-8")
    reg = yaml.safe_load(text)
    for key, entry in reg.items():
        method = entry.get("method")
        if method not in METHODS:
            raise ValueError(f"{key}: unknown method {method!r}; use one of {sorted(METHODS)}")
        missing = ({"name", "sensor"} | METHODS[method]) - set(entry)
        if missing:
            raise ValueError(f"{key}: missing {sorted(missing)}")
    return reg


# -------------------------------------------------------------- what each source offers


def zenodo_files(record: dict) -> list[Remote]:
    """The files of a Zenodo record, from the JSON of /api/records/<id>."""
    files = record.get("files") or []
    if isinstance(files, dict):  # the other API shape: {"entries": [...]}
        files = files.get("entries") or []
    out = []
    for f in files:
        name = f["key"]
        url = (f.get("links") or {}).get("self", "")
        if not url.endswith("/content"):
            url = f"{ZENODO}/records/{record['id']}/files/{quote(name)}?download=1"
        checksum = f.get("checksum") or ""
        md5 = checksum.removeprefix("md5:") if checksum.startswith("md5:") else None
        out.append(Remote(name, url, f.get("size"), md5))
    return out


def links_in(page: str, base: str, pattern: str) -> list[Remote]:
    """Links on a web page whose path matches `pattern`, made absolute, one per file name."""
    wanted = re.compile(pattern, re.IGNORECASE)
    found: dict[str, str] = {}
    for href in re.findall(r"""href\s*=\s*["']([^"']+)["']""", page, re.IGNORECASE):
        url = urljoin(base, html.unescape(href.strip()))
        path = urlparse(url).path
        if wanted.search(path):
            found.setdefault(Path(path).name, url)
    return [Remote(name, url) for name, url in sorted(found.items())]


def head_size(session, url: str) -> int | None:
    """A file's size from the server without downloading it, when the server says."""
    try:
        r = session.head(url, headers=HEADERS, allow_redirects=True, timeout=30)
        size = int(r.headers.get("Content-Length") or 0)
    except (requests.RequestException, ValueError):
        return None
    return size if r.ok and size > 0 else None


def listing(entry: dict, session) -> tuple[list[Remote], dict]:
    """The files a dataset consists of, and the version they were listed at."""
    method = entry["method"]
    if method == "zenodo":
        url = f"{ZENODO}/api/records/{entry['record']}"
        r = session.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 404:  # a concept id: ask for its latest version
            r = session.get(url + "/versions/latest", headers=HEADERS, timeout=60)
        r.raise_for_status()
        record = r.json()
        version = {
            "record": record.get("id"),
            "doi": record.get("doi"),
            "version": (record.get("metadata") or {}).get("version"),
        }
        return zenodo_files(record), version
    if method == "hf":
        from huggingface_hub import HfApi

        info = HfApi().repo_info(
            entry["repo"], repo_type=entry.get("repo_type", "dataset"), files_metadata=True
        )
        files = [
            Remote(s.rfilename, f"hf://{entry['repo']}/{s.rfilename}", s.size)
            for s in info.siblings
            if any(fnmatch.fnmatch(s.rfilename, p) for p in entry["patterns"])
        ]
        return files, {"repo": entry["repo"], "revision": info.sha}
    if method == "page":
        r = session.get(entry["page"], headers=HEADERS, timeout=60)
        r.raise_for_status()
        files = links_in(r.text, r.url, entry["pattern"])  # r.url: after any redirect
        for f in files:
            f.size = head_size(session, f.url)
        return files, {"page": r.url}
    return [], {}


# ---------------------------------------------------------------------------- download


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 * CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(session, remote: Remote, dest: Path, retries: int = 8) -> Path:
    """Download one file, resuming a partial download, and check its size and md5.

    Bytes go to <name>.part, renamed only once complete and checked: a file without
    .part is always whole, and rerunning skips it.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            _download(session, remote, part)
            _check(remote, part)
            part.rename(dest)
            return dest
        except (requests.RequestException, OSError) as err:
            if attempt == retries:
                raise
            wait = min(300, 15 * attempt)
            print(f"   {remote.name}: {err}. Retry {attempt}/{retries - 1} in {wait} s")
            time.sleep(wait)
    return dest


def _download(session, remote: Remote, part: Path) -> None:
    have = part.stat().st_size if part.exists() else 0
    if part.exists() and remote.size is not None and have >= remote.size:
        return  # every byte is here; _check decides whether they are right
    headers = dict(HEADERS)
    if have:
        headers["Range"] = f"bytes={have}-"
    with session.get(remote.url, headers=headers, stream=True, timeout=(30, 300)) as r:
        if have and r.status_code == 416:  # the server has nothing more to send
            return
        r.raise_for_status()
        if have and r.status_code != 206:  # the server ignored the range: start again
            have = 0
        length = int(r.headers.get("Content-Length") or 0)
        total = remote.size or (have + length if length else None)
        with (
            open(part, "ab" if have else "wb") as fh,
            tqdm(
                total=total,
                initial=have,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=remote.name[-40:],
                mininterval=5,
            ) as bar,
        ):
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))


def _check(remote: Remote, part: Path) -> None:
    size = part.stat().st_size
    if remote.size is not None and size < remote.size:
        raise OSError(f"stopped at {size:,} of {remote.size:,} bytes")  # resumable
    too_long = remote.size is not None and size > remote.size
    if too_long or (remote.md5 is not None and md5sum(part) != remote.md5):
        part.unlink()
        raise OSError("size or md5 differs from the source; downloading again")


def still_to_get(files: list[Remote], dest: Path) -> int:
    """Bytes still to download (files of unknown size count as nothing)."""
    left = 0
    for f in files:
        done, part = dest / f.name, dest / (f.name + ".part")
        if done.exists():
            have = done.stat().st_size
        elif part.exists():
            have = part.stat().st_size
        else:
            have = 0
        left += max(0, (f.size or 0) - have)
    return left


def gb(n: int | None) -> str:
    return "?" if n is None else f"{n / 1e9:,.2f} GB"


def write_manifest(dest: Path, key: str, entry: dict, version: dict, files: list[Remote]) -> Path:
    manifest = {
        "dataset": key,
        "name": entry["name"],
        "sensor": entry["sensor"],
        "licence": entry.get("licence"),
        "cite": entry.get("cite"),
        "method": entry["method"],
        "source": version,
        "downloaded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": [
            {
                "name": f.name,
                "bytes": (dest / f.name).stat().st_size,
                "md5": f.md5 or md5sum(dest / f.name),  # worked out here if the source has none
                "url": f.url,
            }
            for f in files
        ],
    }
    path = dest / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return path


# --------------------------------------------------------------------------------- CLI


def run(key: str, entry: dict, root: Path, session, download: bool) -> None:
    dest = root / "raw" / key
    print(f"\n== {key}: {entry['name']}\n   folder: {dest}")
    if entry["method"] == "manual":
        print(f"   by hand: {entry['note']}")
        return
    files, version = listing(entry, session)
    if not files:
        print("   no files found: the source has changed; check the page by hand")
        return
    for f in files[:15]:
        print(f"   {gb(f.size):>12}  {f.name}")
    if len(files) > 15:
        print(f"   {'':>12}  ... and {len(files) - 15} more")
    unknown = sum(f.size is None for f in files)
    total = gb(sum(f.size or 0 for f in files))
    print(f"   {len(files)} files, {total}" + (f" + {unknown} of unknown size" if unknown else ""))
    if not download:
        return
    dest.mkdir(parents=True, exist_ok=True)
    need, free = still_to_get(files, dest), shutil.disk_usage(dest).free
    print(f"   still to download {gb(need)}, free on this disk {gb(free)}")
    if need > 0.95 * free:
        raise SystemExit(f"   not enough space for {key}: free some, or move CATENA_DATA")
    if entry["method"] == "hf":
        from huggingface_hub import snapshot_download

        snapshot_download(
            entry["repo"],
            repo_type=entry.get("repo_type", "dataset"),
            revision=version["revision"],
            local_dir=dest,
            allow_patterns=entry["patterns"],
        )
    else:
        for i, f in enumerate(files, 1):
            print(f"   [{i}/{len(files)}] {f.name}")
            fetch(session, f, dest / f.name)
    print(f"   done. Manifest: {write_manifest(dest, key, entry, version, files)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m catena.data.download", description="Download the cloud datasets (D1)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the datasets in sources.yaml")
    for command, text in [
        ("plan", "list files and sizes, download nothing"),
        ("get", "download into $CATENA_DATA/raw/<key>/"),
    ]:
        cmd = sub.add_parser(command, help=text)
        cmd.add_argument("keys", nargs="+", metavar="dataset")
        cmd.add_argument(
            "--root",
            default=os.environ.get("CATENA_DATA"),
            help="data folder (default: $CATENA_DATA)",
        )
    args = parser.parse_args(argv)
    reg = registry()
    if args.command == "list":
        for key, e in reg.items():
            print(f"{key:26} {e['method']:7} {e['sensor']:14} {e['name']}")
        return
    unknown = [k for k in args.keys if k not in reg]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; run `list` to see the names")
    if not args.root:
        parser.error("set CATENA_DATA (see the setup notes) or pass --root")
    session = requests.Session()
    for key in args.keys:
        run(key, reg[key], Path(args.root).expanduser(), session, download=args.command == "get")


if __name__ == "__main__":
    main()
