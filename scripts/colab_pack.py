"""
Git-based Colab Bundle Packer & Isolated Import Verifier (`scripts/colab_pack.py`).

Builds a slim `.tar.gz` bundle directly from `git ls-files` (tracked + non-ignored
working tree files), filtering out heavy checkpoints/caches (`*.pt`, `*.npz`, `*.bin`,
`*.zip`, `> max_file_mb`), and optionally verifies transitive imports inside a truly
isolated `TemporaryDirectory` (`cwd=td`, workspace root stripped from `sys.path`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT_DIR = Path(__file__).resolve().parents[1]
SKIP_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".bin", ".npz", ".npy", ".ckpt", ".zip", ".pdf", ".xz"}
SKIP_DIR_PARTS = {"__pycache__", ".git", ".pytest_cache", "checkpoints", "archive_v12_v36"}


def collect_bundle_files(root: Path, max_file_bytes: int) -> list[Path]:
    res = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=str(root),
        capture_output=True,
        check=True,
    )
    rel_paths = [p for p in res.stdout.decode("utf-8", errors="replace").split("\0") if p]
    selected: list[Path] = []
    for rel in sorted(set(rel_paths)):
        fp = root / rel
        if not fp.is_file():
            continue
        parts = set(Path(rel).parts)
        if parts & SKIP_DIR_PARTS:
            continue
        if fp.suffix.lower() in SKIP_SUFFIXES:
            continue
        if fp.stat().st_size > max_file_bytes:
            continue
        selected.append(fp)
    return selected


def build_colab_bundle(
    out_path: Path,
    root: Path = ROOT_DIR,
    max_file_mb: float = 5.0,
    verify_imports: list[str] | None = None,
) -> tuple[int, int]:
    files = collect_bundle_files(root, int(max_file_mb * 1024 * 1024))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tf:
        for fp in files:
            tf.add(fp, arcname=fp.relative_to(root).as_posix())
    n_bytes = out_path.stat().st_size
    print(f"[colab_pack] Built {out_path} ({n_bytes:,} B, {len(files)} files)", flush=True)

    if verify_imports:
        with tempfile.TemporaryDirectory(prefix="colab_pack_iso_") as td:
            with tarfile.open(out_path, "r:gz") as tf:
                tf.extractall(td, filter="data")
            mod_list = repr(list(verify_imports))
            chk = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-c",
                    (
                        "import importlib, pathlib, sys; "
                        f"root = pathlib.Path({td!r}); "
                        f"repo_lc = {str(root).lower()!r}; "
                        "sys.path = [str(root)] + [p for p in sys.path if p and repo_lc not in p.lower()]; "
                        f"mods = {mod_list}; "
                        "loaded = [importlib.import_module(m).__file__ for m in mods]; "
                        "assert all(str(root).lower() in str(f).lower() for f in loaded), loaded; "
                        "print('ISOLATED_IMPORT_OK', len(loaded), loaded[0])"
                    ),
                ],
                cwd=td,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if chk.returncode != 0:
                raise RuntimeError(f"Isolated bundle verification failed:\n{chk.stdout}\n{chk.stderr}")
            print(f"[colab_pack] {chk.stdout.strip()}", flush=True)
    return n_bytes, len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack slim Colab bundle from git ls-files with isolated import verification.")
    parser.add_argument("--out", required=True, help="Output .tar.gz path")
    parser.add_argument("--max-file-mb", type=float, default=5.0, help="Skip files larger than this size in MB")
    parser.add_argument(
        "--verify-import",
        action="append",
        default=[],
        help="Module name to import inside isolated extracted tempdir (repeatable)",
    )
    args = parser.parse_args()
    build_colab_bundle(Path(args.out), ROOT_DIR, max_file_mb=args.max_file_mb, verify_imports=args.verify_import)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
