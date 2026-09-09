#!/usr/bin/env python3
"""Fetch the original *Lost in the Middle* data from GitHub.

The data is ~257 MB gzipped and is **not** committed to this repository. It is
downloaded from the upstream repo at a pinned commit and verified against SHA-256
digests recorded in :data:`litm2026.data.DATA_MANIFEST`.

Two sources, in preference order:

``--method raw`` (default)
    ``https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/<commit>/<path>``
    Fetches only the files you ask for. This is the cheap option when you only need
    the 20-document sweep.

``--method git``
    ``git clone --filter=blob:none --no-checkout`` at the pinned commit, then a
    sparse checkout of the two data directories. Use this if raw.githubusercontent
    is blocked on your network.

Note: the paper's README also documents ``wget https://nlp.stanford.edu/data/...``
for the *raw Contriever retrieval results*, which are the inputs used to
*regenerate* the QA data. This replication deliberately does not regenerate
anything -- regenerating the data would make it a different experiment -- so that
host is not needed and is not contacted.

Examples::

    # Everything (~257 MB)
    python scripts/download_data.py

    # Just what the main QA config needs (~76 MB)
    python scripts/download_data.py --subset qa20

    # Verify an existing download without re-fetching
    python scripts/download_data.py --verify-only

    # Copy from a clone you already have
    python scripts/download_data.py --from-local /path/to/lost-in-the-middle
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from typing import Dict, List, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from litm2026.data import (  # noqa: E402
    DATA_MANIFEST,
    UPSTREAM_COMMIT,
    UPSTREAM_REPO,
    DataFileSpec,
    sha256_file,
)

#: Named groups so you can download only what a given config needs.
SUBSETS: Dict[str, List[str]] = {
    "all": sorted(DATA_MANIFEST),
    "oracle": ["qa_data/nq-open-oracle.jsonl.gz"],
    "qa10": sorted(k for k in DATA_MANIFEST if "10_total_documents" in k)
    + ["qa_data/nq-open-oracle.jsonl.gz"],
    "qa20": sorted(k for k in DATA_MANIFEST if "20_total_documents" in k)
    + ["qa_data/nq-open-oracle.jsonl.gz"],
    "qa30": sorted(k for k in DATA_MANIFEST if "30_total_documents" in k)
    + ["qa_data/nq-open-oracle.jsonl.gz"],
    "kv": sorted(k for k in DATA_MANIFEST if k.startswith("kv_retrieval_data/")),
}


def human(num_bytes: int) -> str:
    """Format a byte count for humans."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"  # pragma: no cover


def verify(path: pathlib.Path, spec: DataFileSpec) -> bool:
    """Check one file's size and SHA-256 against the manifest."""
    if not path.exists():
        return False
    if path.stat().st_size != spec.size_bytes:
        return False
    return sha256_file(path) == spec.sha256


def download_one(spec: DataFileSpec, dest: pathlib.Path, *, force: bool) -> bool:
    """Download and verify a single data file. Returns True on success."""
    target = dest / spec.relpath
    if not force and verify(target, spec):
        print(f"  ok (cached)  {spec.relpath}")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching     {spec.relpath}  ({human(spec.size_bytes)})", flush=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(spec.raw_url, timeout=120) as response, open(tmp, "wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"  FAILED       {spec.relpath}: {exc}", file=sys.stderr)
        return False
    tmp.replace(target)
    if not verify(target, spec):
        print(
            f"  CHECKSUM MISMATCH for {spec.relpath}. The file was removed.\n"
            "  This means the upstream file changed or the download was corrupted. "
            "Do not run experiments on unverified data.",
            file=sys.stderr,
        )
        target.unlink(missing_ok=True)
        return False
    print(f"  verified     {spec.relpath}")
    return True


def copy_from_local(source: pathlib.Path, dest: pathlib.Path, relpaths: Sequence[str]) -> int:
    """Copy data out of an existing clone, verifying every file."""
    failures = 0
    for rel in relpaths:
        spec = DATA_MANIFEST[rel]
        src = source / rel
        if not src.exists():
            print(f"  missing in source: {rel}", file=sys.stderr)
            failures += 1
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        if verify(target, spec):
            print(f"  verified     {rel}")
        else:
            print(f"  CHECKSUM MISMATCH after copy: {rel}", file=sys.stderr)
            failures += 1
    return failures


def clone_via_git(dest: pathlib.Path, relpaths: Sequence[str]) -> int:
    """Sparse-clone the upstream repo at the pinned commit and copy the data out."""
    with tempfile.TemporaryDirectory() as tmpdir:
        work = pathlib.Path(tmpdir) / "lost-in-the-middle"
        commands = [
            ["git", "clone", "--filter=blob:none", "--no-checkout", UPSTREAM_REPO + ".git", str(work)],
            ["git", "-C", str(work), "sparse-checkout", "init", "--cone"],
            ["git", "-C", str(work), "sparse-checkout", "set", "qa_data", "kv_retrieval_data"],
            ["git", "-C", str(work), "checkout", UPSTREAM_COMMIT],
        ]
        for command in commands:
            print("  $ " + " ".join(command), flush=True)
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stderr.strip(), file=sys.stderr)
                return 1
        return copy_from_local(work, dest, relpaths)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dest", type=pathlib.Path, default=REPO_ROOT / "data" / "original", help="Where to put the data."
    )
    parser.add_argument("--subset", choices=sorted(SUBSETS), default="all", help="Which files to fetch.")
    parser.add_argument("--method", choices=("raw", "git"), default="raw", help="How to fetch.")
    parser.add_argument("--from-local", type=pathlib.Path, help="Copy from an existing clone instead.")
    parser.add_argument("--verify-only", action="store_true", help="Check what is present; download nothing.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file verifies.")
    args = parser.parse_args(argv)

    relpaths = sorted(set(SUBSETS[args.subset]))
    total = sum(DATA_MANIFEST[r].size_bytes for r in relpaths)
    dest = args.dest
    print(f"Upstream: {UPSTREAM_REPO} @ {UPSTREAM_COMMIT}")
    print(f"Files: {len(relpaths)} ({human(total)} gzipped)   ->  {dest}\n")

    if args.verify_only:
        missing = 0
        for rel in relpaths:
            state = "ok" if verify(dest / rel, DATA_MANIFEST[rel]) else "MISSING or CORRUPT"
            missing += state != "ok"
            print(f"  {state:<20} {rel}")
        print(f"\n{len(relpaths) - missing}/{len(relpaths)} file(s) verified.")
        return 0 if missing == 0 else 1

    if args.from_local:
        failures = copy_from_local(args.from_local, dest, relpaths)
    elif args.method == "git":
        failures = clone_via_git(dest, relpaths)
    else:
        failures = sum(0 if download_one(DATA_MANIFEST[r], dest, force=args.force) else 1 for r in relpaths)

    if failures:
        print(f"\n{failures} file(s) failed. Retry, or use --method git.", file=sys.stderr)
        return 1
    print(f"\nAll {len(relpaths)} file(s) present and verified in {dest}.")
    print("Next: python -m litm2026.runner --config experiments/configs/pilot.yaml --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
