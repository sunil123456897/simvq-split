"""
Automated Repository Guardrail (`scripts/repo_guardrail.py`).

Shared deterministic checks used by `.git/hooks/pre-commit` and phase validators:
1. Zero diff on untouched `simvq_split/baselines.py` vs `HEAD`.
2. Ruff syntax/fatal lint check (`E9,F63,F7,F82`) across core package, kernels, tests, scripts, and runners.
3. 17-law Zero-Cheat `pytest tests/test_anti_cheat.py -q` + v3 CUDA/QAT `tests/test_fast_calcite_scans.py -q`.
4. `AGENTS.md` size cap (`<= 24,000 B`, prevents silent context truncation).
5. UTF-8 (no BOM / U+FFFD) and Russian lexical quarantine across repository Markdown docs & workspace skills.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

RUFF_TARGETS = (
    "simvq_split",
    "calcite_v2",
    "kernels",
    "tests",
    "scripts",
    "experiments",
    "run_simvq_split_benchmark.py",
    "validate_simvq_split_results.py",
    "run_phase_b_byte_benchmark.py",
    "validate_phase_b_byte_results.py",
    "run_phase_e_fast_utf8_benchmark.py",
    "validate_phase_e_results.py",
    "run_phase_f_utf8_cram_benchmark.py",
    "validate_phase_f_results.py",
    "validate_phase_g_results.py",
    "validate_phase_h_results.py",
    "validate_phase_i_results.py",
    "run_phase_j_uebr_benchmark.py",
    "validate_phase_j_results.py",
    "validate_phase_k_results.py",
)

OPTIONAL_RUFF_TARGETS = ()

FORBIDDEN_PHRASES = (
    "не просто",
    "является",
    "являются",
    "осуществлять",
    "осуществляется",
    "в рамках",
    "симфония",
    "гобелен",
    "квантовый скачок",
    "бесшовная экосистема",
    "уникальный опыт",
    "феноменальный",
    "колоссальный",
    "в заключение",
    "таким образом",
)


AGENTS_MD_MAX_BYTES = 24_000


def collect_guarded_markdown_paths(root: Path) -> list[Path]:
    """Return existing repository Markdown documents governed by UTF-8 and lexical quarantine."""
    candidates = [
        root / "README.md",
        root / "SimVQ_Split_Benchmark_Report.md",
        root / "Phase_B_Byte_Chunker_Report.md",
        root / "Phase_E_Fast_UTF8_Report.md",
        root / "Phase_F_UTF8_Iterative_CRAM_Report.md",
        root / "Phase_G_Colab_ScaleUp_Report.md",
        root / "Phase_H_Tier1_QAT_Closure_Report.md",
        root / "Phase_J_UEBR_System2_Report.md",
        root / "Phase_K_System1_Scaling_Report.md",
        root / "Phase_L_Calcite_v2_Report.md",
        root / "Phase_M_Calcite_v3_Report.md",
        root / "Post_Transformer_Architecture_v5_Blueprint.md",
        root / "AGENTS.md",
        root / "CODING_STANDARDS.md",
    ]
    synth_dir = root / "research" / "post_transformer_synthesis"
    if synth_dir.is_dir():
        candidates.extend(sorted(synth_dir.glob("*.md")))
    skill_dir = root / ".agents" / "skills" / "calcite-phase-runner"
    if skill_dir.is_dir():
        candidates.extend(sorted(skill_dir.rglob("*.md")))
    return [p for p in candidates if p.exists()]


def check_agents_md_size(root: Path, max_bytes: int = AGENTS_MD_MAX_BYTES) -> int:
    """Assert `AGENTS.md` stays within the 24 KB rule context cap."""
    p = root / "AGENTS.md"
    size = p.stat().st_size
    if size > max_bytes:
        raise AssertionError(
            f"`AGENTS.md` exceeds {max_bytes:,} B rule cap ({size:,} B); condense or move archive pointers to references/."
        )
    return size


def check_baselines_untouched(root: Path) -> None:
    """Assert `simvq_split/baselines.py` has a zero-byte git diff against `HEAD`."""
    res = subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "simvq_split/baselines.py"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        detail = (res.stdout + "\n" + res.stderr).strip()
        raise AssertionError(f"`simvq_split/baselines.py` must stay untouched vs HEAD!\n{detail}")


def check_ruff_lint(root: Path) -> None:
    """Run ruff fatal lint checks (`E9,F63,F7,F82`) across Python sources."""
    targets = list(RUFF_TARGETS) + [t for t in OPTIONAL_RUFF_TARGETS if (root / t).exists()]
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "E9,F63,F7,F82",
            *targets,
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        detail = (res.stdout + "\n" + res.stderr).strip()
        raise AssertionError(f"Ruff check failed!\n{detail}")


def check_anti_cheat_pytest(root: Path) -> str:
    """Run `pytest tests/test_anti_cheat.py -q` and require at least 17 passing laws."""
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(root / "tests" / "test_anti_cheat.py"),
            "-q",
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    m = re.search(r"(\d+)\s+passed", res.stdout)
    if res.returncode != 0 or m is None or int(m.group(1)) < 17 or "failed" in res.stdout:
        detail = (res.stdout + "\n" + res.stderr).strip()
        raise AssertionError(f"Zero-Cheat pytest suite (>= 17 laws) failed!\n{detail}")
    return res.stdout.strip()


def check_markdown_quarantine(root: Path, paths: list[Path] | None = None) -> int:
    """Verify UTF-8 integrity (no BOM, no U+FFFD) and zero forbidden clichés across docs."""
    targets = collect_guarded_markdown_paths(root) if paths is None else paths
    for p in targets:
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM detected in {p}"
        text = raw.decode("utf-8")
        assert "\ufffd" not in text, f"UTF-8 replacement char found in {p}"
        low = text.lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in low, f"Forbidden cliche `{phrase}` found in {p.name}!"
    return len(targets)


def check_fast_kernels_pytest(root: Path) -> str:
    """Run `pytest tests/test_fast_calcite_scans.py -q` (v3 CUDA/QAT + `AdamW(fused=True)` regression)."""
    test_p = root / "tests" / "test_fast_calcite_scans.py"
    if not test_p.exists():
        return "skipped (not present)"
    res = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_p), "-q"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    m = re.search(r"(\d+)\s+passed", res.stdout)
    if res.returncode != 0 or m is None or int(m.group(1)) < 5 or "failed" in res.stdout:
        detail = (res.stdout + "\n" + res.stderr).strip()
        raise AssertionError(f"v3 CUDA/QAT pytest suite (>= 5 tests) failed!\n{detail}")
    return res.stdout.strip()


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    check_baselines_untouched(root)
    print("OK baselines: 0-byte diff vs HEAD", flush=True)
    check_ruff_lint(root)
    print("OK ruff: E9,F63,F7,F82 clean", flush=True)
    pytest_out = check_anti_cheat_pytest(root)
    print(f"OK pytest: {pytest_out.splitlines()[-1]}", flush=True)
    fk_out = check_fast_kernels_pytest(root)
    print(f"OK fast_kernels: {fk_out.splitlines()[-1]}", flush=True)
    agents_b = check_agents_md_size(root)
    print(f"OK AGENTS.md: {agents_b:,} B <= {AGENTS_MD_MAX_BYTES:,} B", flush=True)
    n_docs = check_markdown_quarantine(root)
    print(f"OK quarantine: {n_docs} Markdown files verified clean", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
