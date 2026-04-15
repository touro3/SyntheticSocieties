from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from decision.dataset_router import DatasetRegistry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream-download datasets from the registry without loading them into memory."
    )
    parser.add_argument("--registry-path", type=str, default="data/dataset_registry.json")
    parser.add_argument(
        "--dataset-ids", type=str, default=None, help="Comma-separated dataset ids. Default: all with download_url."
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-mb", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_stream(url: str, dest: Path, chunk_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def main():
    args = parse_args()
    registry = DatasetRegistry(args.registry_path)
    requested = None
    if args.dataset_ids:
        requested = {x.strip() for x in args.dataset_ids.split(",") if x.strip()}

    audit_path = Path("data") / "dataset_sync_audit.jsonl"
    chunk_size = args.chunk_mb * 1024 * 1024

    for ds in registry.list():
        if requested and ds["id"] not in requested:
            continue
        if not ds.get("download_url"):
            continue

        dest = Path(ds["local_path"])
        if dest.exists() and not args.force:
            status = "skipped_exists"
        else:
            download_stream(ds["download_url"], dest, chunk_size)
            status = "downloaded"

        checksum = sha256_file(dest) if dest.exists() else None
        if ds.get("sha256") and checksum != ds["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {ds['id']}: expected {ds['sha256']}, got {checksum}")

        record = {
            "dataset_id": ds["id"],
            "status": status,
            "local_path": str(dest),
            "sha256": checksum,
            "download_url": ds.get("download_url"),
        }
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(record)


if __name__ == "__main__":
    main()
