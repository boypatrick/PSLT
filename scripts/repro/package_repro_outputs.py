#!/usr/bin/env python3
"""
Package paper reproducibility artifacts into a normalized run directory.

Input mapping:
  repro/artifact_map.csv

Output layout:
  <run_dir>/
    figures/
    tables/
    paper/
    artifact_index.csv
    checksums.sha256
    manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


KIND_DIR = {
    "figure": "figures",
    "table": "tables",
    "paper": "paper",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


def resolve_source(root: Path, pattern: str) -> Optional[Path]:
    if has_glob(pattern):
        matches = sorted(root.glob(pattern))
    else:
        p = root / pattern
        matches = [p] if p.exists() else []
    if not matches:
        return None
    matches.sort(key=lambda p: (p.stat().st_mtime, p.as_posix()))
    return matches[-1]


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def main() -> int:
    here = Path(__file__).resolve()
    default_root = here.parents[2]
    default_map = default_root / "repro" / "artifact_map.csv"

    ap = argparse.ArgumentParser(description="Package reproducibility artifacts.")
    ap.add_argument("--root", default=str(default_root), help="Repository root.")
    ap.add_argument("--run-dir", required=True, help="Run directory for packaged artifacts.")
    ap.add_argument("--map-file", default=str(default_map), help="Artifact mapping CSV.")
    ap.add_argument(
        "--allow-missing-required",
        action="store_true",
        help="Do not fail when required artifacts are missing.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve()
    map_file = Path(args.map_file).resolve()

    if not map_file.exists():
        print(f"[error] map file not found: {map_file}", file=sys.stderr)
        return 2

    run_dir.mkdir(parents=True, exist_ok=True)
    for _, subdir in KIND_DIR.items():
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    with map_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    index_rows: List[Dict[str, str]] = []
    checksum_lines: List[str] = []
    missing_required: List[Dict[str, str]] = []
    missing_optional: List[Dict[str, str]] = []

    for row in rows:
        kind = row["kind"].strip()
        if kind not in KIND_DIR:
            print(f"[error] invalid kind '{kind}' in map file", file=sys.stderr)
            return 2

        source_pattern = row["source"].strip()
        artifact_name = row["artifact_name"].strip()
        required = parse_bool(row.get("required", "0"))
        source_path = resolve_source(root, source_pattern)

        if source_path is None:
            status = "missing_required" if required else "missing_optional"
            rec = {
                "id": row.get("id", ""),
                "kind": kind,
                "source": source_pattern,
                "artifact_name": artifact_name,
                "status": status,
                "note": row.get("notes", ""),
            }
            if required:
                missing_required.append(rec)
            else:
                missing_optional.append(rec)
            index_rows.append(
                {
                    "id": rec["id"],
                    "kind": kind,
                    "status": status,
                    "source_pattern": source_pattern,
                    "resolved_source": "",
                    "artifact_name": artifact_name,
                    "artifact_relpath": "",
                    "sha256": "",
                    "size_bytes": "",
                    "note": rec["note"],
                }
            )
            continue

        out_dir = run_dir / KIND_DIR[kind]
        out_path = out_dir / artifact_name
        shutil.copy2(source_path, out_path)
        digest = sha256sum(out_path)
        size_bytes = out_path.stat().st_size
        rel_out = out_path.relative_to(run_dir).as_posix()
        checksum_lines.append(f"{digest}  {rel_out}")
        index_rows.append(
            {
                "id": row.get("id", ""),
                "kind": kind,
                "status": "copied",
                "source_pattern": source_pattern,
                "resolved_source": source_path.relative_to(root).as_posix(),
                "artifact_name": artifact_name,
                "artifact_relpath": rel_out,
                "sha256": digest,
                "size_bytes": str(size_bytes),
                "note": row.get("notes", ""),
            }
        )

    index_path = run_dir / "artifact_index.csv"
    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "kind",
                "status",
                "source_pattern",
                "resolved_source",
                "artifact_name",
                "artifact_relpath",
                "sha256",
                "size_bytes",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    checksums_path = run_dir / "checksums.sha256"
    with checksums_path.open("w") as f:
        if checksum_lines:
            f.write("\n".join(checksum_lines) + "\n")

    manifest = {
        "generated_at_utc": utc_now_iso(),
        "repo_root": str(root),
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "git_commit": git_commit(root),
        "map_file": str(map_file),
        "counts": {
            "map_rows": len(rows),
            "copied": sum(1 for r in index_rows if r["status"] == "copied"),
            "missing_required": len(missing_required),
            "missing_optional": len(missing_optional),
        },
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[saved] {index_path}")
    print(f"[saved] {checksums_path}")
    print(f"[saved] {manifest_path}")
    print(
        "[summary]",
        f"copied={manifest['counts']['copied']}",
        f"missing_required={manifest['counts']['missing_required']}",
        f"missing_optional={manifest['counts']['missing_optional']}",
    )

    if missing_required and not args.allow_missing_required:
        print("[error] missing required artifacts detected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
