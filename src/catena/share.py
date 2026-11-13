"""Keeping the work: HEO's and Eaglet II's data in a private Hugging Face repository, the rest
in a public one and in a Zenodo draft.

    python -m catena.share DATA [--private] [--public] [--zenodo] [--make-public] [--dry-run]

  --private   HEO's and Eaglet II's data (PRIVATE: their products, labels and HEO's converted
              bands) to the dataset repository <user>/catena-private, as the folders are, so
              a download put back under DATA restores them. These scenes may not be published:
              before any upload the repository must say it is private, else nothing is sent.
  --public    Phi-sat-2's products and labels, the trained models (final.pt with its
              config.json, of the runs the paper uses: MODEL_RUNS) and every result (the CSVs,
              JSON, logs and metrics under DATA/runs; no image, no light.csv, which places HEO's
              and Eaglet II's scenes) packed into ARCHIVES in --stage, with a README.md, and
              uploaded to <user>/catena. It stays private until you make it public on its
              page (Settings), or is made public at once with --make-public.
  --zenodo    the same archives to a new Zenodo draft. A published Zenodo record can never be
              deleted, so this script only makes the draft: its metadata (title, authors,
              licence) and the Publish button are yours, on zenodo.org.
  --dry-run   what would go where, with sizes; nothing packed, nothing sent.
The public datasets (CloudSEN12+, L8 Biome, L7 Irish, CloudPeru2) go nowhere: their owners
publish them, and catena's scripts rebuild the chips from them.

Hugging Face: pip install -U huggingface_hub, then hf auth login (the token is kept in
~/.cache/huggingface). Zenodo: a token with the scopes deposit:write and deposit:actions, in
ZENODO_TOKEN or in ~/.zenodo_token. An interrupted upload goes on where it stopped when the
same command is run again (huggingface_hub keeps its progress in .cache/huggingface inside
the folder it uploads).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

PRIVATE = (  # may not be published: kept for you alone
    "raw/iride_heo",
    "raw/iride_eaglet2",
    "labels/iride_heo",
    "labels/iride_eaglet2",
    "chips/iride_heo",
)
PHISAT2 = ("raw/phisat2", "labels/phisat2")
MODEL_RUNS = ("big", "arms", "nas_final", "delivered", "nas_coarser", "flight")
RESULTS = ("*.csv", "*.json", "*.jsonl", "*.log", "*.txt")
WITHHELD = ("light.csv",)  # results that place HEO's and Eaglet II's scenes: kept private
ARCHIVES = ("phisat2.tar", "models.tar", "results.tar")
ZENODO = "https://zenodo.org"
GB = 1e9


def files_under(data: Path, folders) -> list[Path]:
    """Every file under DATA's folders (paths relative to DATA, sorted); none of a cache."""
    found = []
    for folder in folders:
        root = data / folder
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            relative = path.relative_to(data)
            if path.is_file() and ".cache" not in relative.parts:
                found.append(relative)
    return sorted(found)


def public_files(data: Path) -> dict[str, list[Path]]:
    """What goes in each public archive, paths relative to DATA."""
    runs = data / "runs"
    models = []
    for run in MODEL_RUNS:
        for final in sorted((runs / run).rglob("final.pt")) if (runs / run).is_dir() else []:
            models.append(final.relative_to(data))
            config = final.with_name("config.json")
            if config.exists():
                models.append(config.relative_to(data))
    results = [
        p
        for p in files_under(data, ["runs"])
        if any(fnmatch.fnmatch(p.name, pattern) for pattern in RESULTS) and p.name not in WITHHELD
    ]
    out = {"phisat2.tar": files_under(data, PHISAT2), "models.tar": sorted(models)}
    return out | {"results.tar": results}


def size(data: Path, paths) -> int:
    return sum((data / p).stat().st_size for p in paths)


def check_apart(archives: dict) -> None:
    """Nothing of PRIVATE may end up in a public archive."""
    for name, paths in archives.items():
        for p in paths:
            if any(str(p) == f or str(p).startswith(f + "/") for f in PRIVATE):
                raise SystemExit(f"{p} is HEO's or Eaglet II's and was headed for {name}: stopped")


def readme(archives: dict, data: Path) -> str:
    lines = [
        "# Catena: sensor-agnostic cloud and cloud-shadow masking",
        "",
        "Public part of the data, models and results. HEO's and Eaglet II's scenes and labels",
        "are not here: they may not be published (available from their owners on request).",
        "",
        "| Archive | Files | Size | What |",
        "|---|---|---|---|",
    ]
    what = {
        "phisat2.tar": "Phi-sat-2: the products and their reviewed labels (raw/, labels/)",
        "models.tar": "the trained networks: final.pt and config.json of each run and seed",
        "results.tar": "every score (CSV), configuration, log and training curve of the runs",
    }
    for name, paths in archives.items():
        mb = size(data, paths) / 1e6
        lines.append(f"| {name} | {len(paths)} | {mb:,.0f} MB | {what[name]} |")
    lines += [
        "",
        "Unpack each archive inside one folder (DATA): the paths are those catena reads.",
        "The public datasets (CloudSEN12+, L8 Biome, L7 Irish, CloudPeru2) come from their",
        "own publishers; catena's scripts convert them.",
        "",
    ]
    return "\n".join(lines)


def pack(data: Path, archives: dict, stage: Path) -> list[Path]:
    """The archives (plain tar, the paths relative to DATA, in a fixed order, so the same
    files make the same archive) and README.md in stage/public."""
    public = stage / "public"
    public.mkdir(parents=True, exist_ok=True)
    need = sum(size(data, paths) for paths in archives.values())
    free = shutil.disk_usage(public).free
    if need + 1 * GB > free:
        raise SystemExit(f"{need / GB:.1f} GB to pack, {free / GB:.1f} GB free in {public}")
    made = []
    for name, paths in archives.items():
        with tarfile.open(public / name, "w") as tar:
            for p in paths:
                tar.add(data / p, arcname=str(p), recursive=False)
        made.append(public / name)
    (public / "README.md").write_text(readme(archives, data))
    return [*made, public / "README.md"]


def hub():
    """huggingface_hub's API, logged in, and the user's name."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("pip install -U huggingface_hub, then hf auth login") from None
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as e:
        raise SystemExit(f"not logged in to Hugging Face (hf auth login): {e}") from None
    return api, user


def ensure_private(api, repo: str) -> None:
    """The repository exists and is private, or nothing is sent."""
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise SystemExit(f"{repo} is public: HEO's and Eaglet II's data not sent")


def upload_private(api, repo: str, data: Path, workers: int) -> None:
    ensure_private(api, repo)
    patterns = [f"{folder}/*" for folder in PRIVATE]
    api.upload_large_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(data),
        allow_patterns=patterns,
        num_workers=workers,
    )
    ensure_private(api, repo)  # still private once done


def upload_public(api, repo: str, stage: Path, make_public: bool, workers: int) -> None:
    api.create_repo(repo, repo_type="dataset", private=not make_public, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo, repo_type="dataset", folder_path=str(stage / "public"), num_workers=workers
    )


def zenodo_token() -> str:
    token = os.environ.get("ZENODO_TOKEN", "")
    stored = Path.home() / ".zenodo_token"
    if not token and stored.exists():
        token = stored.read_text().strip()
    if not token:
        raise SystemExit("no Zenodo token: put it in ZENODO_TOKEN or in ~/.zenodo_token")
    return token


def call(method: str, url: str, token: str, body=None, path: Path | None = None):
    """One call to Zenodo's REST API: a JSON body, or a file streamed; the JSON answer."""
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "catena-share"}
    if path is not None:
        headers |= {"Content-Type": "application/octet-stream"}
        headers |= {"Content-Length": str(path.stat().st_size)}
        with open(path, "rb") as f:
            request = urllib.request.Request(url, data=f, method=method, headers=headers)
            with urllib.request.urlopen(request, timeout=600) as answer:
                return json.load(answer)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=600) as answer:
        return json.load(answer)


def zenodo_draft(files: list[Path], token: str, host: str = ZENODO) -> str:
    """A new Zenodo draft holding the files; its page (published only by hand)."""
    draft = call("POST", f"{host}/api/deposit/depositions", token, body={})
    bucket = draft["links"]["bucket"]
    for path in files:
        call("PUT", f"{bucket}/{path.name}", token, path=path)
        print(f"  zenodo: {path.name} uploaded", flush=True)
    return draft["links"].get("html", f"{host}/deposit/{draft['id']}")


def report(data: Path, private: list[Path], archives: dict) -> None:
    print(f"private (HEO, Eaglet II): {len(private)} files, {size(data, private) / GB:.1f} GB")
    for folder in PRIVATE:
        mine = [p for p in private if str(p).startswith(folder + "/")]
        print(f"  {folder:<22}{len(mine):>7} files {size(data, mine) / GB:7.2f} GB")
    print("public:")
    for name, paths in archives.items():
        print(f"  {name:<22}{len(paths):>7} files {size(data, paths) / GB:7.2f} GB")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data", type=Path)
    p.add_argument("--private", action="store_true", help="HEO and Eaglet II: private repository")
    p.add_argument("--public", action="store_true", help="the rest: the public repository")
    p.add_argument("--zenodo", action="store_true", help="the public archives: a Zenodo draft")
    p.add_argument("--make-public", action="store_true", help="the public repository public now")
    p.add_argument("--dry-run", action="store_true", help="say what would go where; send nothing")
    p.add_argument(
        "--stage", type=Path, help="where the archives are packed (DATA/../catena_share)"
    )
    p.add_argument("--private-repo", help="default <user>/catena-private")
    p.add_argument("--public-repo", help="default <user>/catena")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    if not (args.private or args.public or args.zenodo):
        p.error("say what to send: --private, --public, --zenodo")
    data = args.data.resolve()
    stage = args.stage or data.parent / "catena_share"
    private = files_under(data, PRIVATE)
    archives = public_files(data)
    check_apart(archives)
    report(data, private, archives)
    if args.dry_run:
        return
    api = user = None
    if args.private or args.public:
        api, user = hub()
    if args.private:
        repo = args.private_repo or f"{user}/catena-private"
        print(f"private: to {repo}", flush=True)
        upload_private(api, repo, data, args.workers)
        print(f"private: done, {repo} is private", flush=True)
    files = []
    if args.public or args.zenodo:
        files = pack(data, archives, stage)
        print(f"packed: {', '.join(f.name for f in files)} in {stage / 'public'}", flush=True)
    if args.public:
        repo = args.public_repo or f"{user}/catena"
        print(f"public: to {repo}", flush=True)
        upload_public(api, repo, stage, args.make_public, args.workers)
        state = "public" if args.make_public else "private until you make it public"
        print(f"public: done, {repo} ({state})", flush=True)
    if args.zenodo:
        page = zenodo_draft(files, zenodo_token())
        print(f"zenodo: a draft, not published: {page}", flush=True)


if __name__ == "__main__":
    main()
