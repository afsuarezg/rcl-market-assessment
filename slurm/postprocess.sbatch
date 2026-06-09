#!/bin/bash
#
# =============================================================================
# Post-processing: build the analysis/ and graphs/ folders on Sherlock for the
# BLP (1995) and Nevo (2000a) multistart sweeps.
# =============================================================================
#
# What this does, in one paragraph:
#   The estimation jobs (blp_blp.py / nevo_blp.py) leave behind the per-spec
#   multistart CSVs under
#       /oak/stanford/groups/polinsky/blp_nevo/results/blp/csv/
#       /oak/stanford/groups/polinsky/blp_nevo/results/nevo/csv/
#   but they do NOT run the two post-estimation modules. This script runs them,
#   in any order (they are independent), against that existing OAK tree:
#       0) git pull --ff-only      -> sync code first (skip with GIT_PULL=0)
#       1) analyze_results.py      -> results/<ds>/analysis/NN_*.txt
#       2) plot_results.py         -> results/<ds>/graphs/NN_*.png
#   Both modules read   results/<ds>/csv/{multistart_all,elasticities_detail}.csv
#   and write back IN PLACE next to where they found the inputs.
#
#   Only blp/ and nevo/ are touched. The blp_non_optimal/ and nevo_non_optimal/
#   folders shown in the OAK results/ directory are NOT handled by these two
#   scripts and are left untouched.
#
# THE ONE GOTCHA — where the outputs land:
#   Neither script takes a --root/--output flag (no argparse). They locate the
#   input CSVs by checking, in order:
#       1. <repo>/results/<ds>/...            (_LOCAL_ROOT, i.e. PROJECT_DIR/results)
#       2. /oak/stanford/groups/polinsky/blp_nevo/results/<ds>/...   (_OAK_ROOT)
#   and write analysis/ + graphs/ NEXT TO wherever the inputs were found. The
#   normal case: the repo has no local results/ (fresh clone — results/ is
#   gitignored), so the scripts fall through to the OAK path and write there.
#   (If the repo happens to be checked out AT the OAK path, PROJECT_DIR/results
#   already IS the OAK results dir — also fine.) A stale/partial local results/
#   would shadow the OAK data and send outputs to the wrong place. The
#   PRECONDITION below resolves and prints the exact target dirs and ABORTS if
#   they are not under OAK, so a misconfiguration is caught before any work runs.
#
# How to submit:
#   From a shell, in your checkout:   cd <repo> && sbatch slurm/postprocess.sbatch
#   Via OnDemand Job Composer:        works as-is IF the repo is cloned at
#                                     $HOME/rcl-market-assessment (see PROJECT_DIR
#                                     resolution below); otherwise set PROJECT_DIR.
#
#   This operates IN PLACE on the existing OAK results/ tree. It runs
#   `git pull --ff-only` first to sync the committed code (skippable with
#   GIT_PULL=0). The pull is non-fatal: if PROJECT_DIR is not a git checkout it
#   is skipped with a warning instead of killing the job. results/ is gitignored,
#   so the pull never touches generated data.
# =============================================================================

#SBATCH --job-name=blp-nevo-postproc        # name shown in `squeue`
#SBATCH --output=blp_nevo_postproc.%j.out   # stdout file (%j = SLURM job id)
#SBATCH --error=blp_nevo_postproc.%j.err    # stderr file
#SBATCH --time=01:00:00                     # max wallclock (read CSVs + plot only)
#SBATCH -p normal                           # partition / queue
#SBATCH -c 4                                # CPU cores (single node)
#SBATCH --mem=16GB                          # total RAM

set -euo pipefail

# Where the repo working copy lives on Sherlock. Resolved in priority order:
#   1. An explicit PROJECT_DIR (env or --export) always wins.
#   2. The submit dir, but ONLY if it's actually the repo. This is true for
#      `cd <checkout> && sbatch slurm/postprocess.sbatch`, but NOT for OnDemand
#      Job Composer, which copies the script into a per-job sandbox
#      (~/ondemand/data/sys/myjobs/projects/...) and submits from there — so
#      SLURM_SUBMIT_DIR is that throwaway dir, with no pyproject.toml.
#   3. The conventional clone location $HOME/rcl-market-assessment — so a plain
#      `git clone` in $HOME + submit-from-anywhere (incl. Job Composer) works
#      with no edits.
# The repo can live anywhere: the scripts find the input CSVs via their built-in
# OAK fallback regardless of where the code lives, as long as no local results/
# shadows it (a fresh clone has none — results/ is gitignored). The precondition
# below confirms outputs will land on OAK before any work runs. Override with:
#     sbatch --export=PROJECT_DIR=$HOME/rcl-market-assessment slurm/postprocess.sbatch
if [ -n "${PROJECT_DIR:-}" ]; then
    :                                                       # explicit override wins
elif [ -f "${SLURM_SUBMIT_DIR:-/nonexistent}/pyproject.toml" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"                          # submitted from the repo
elif [ -f "$HOME/rcl-market-assessment/pyproject.toml" ]; then
    PROJECT_DIR="$HOME/rcl-market-assessment"                # the conventional clone
else
    PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"                # let the guard report it
fi

cd "$PROJECT_DIR"

# Sanity: PROJECT_DIR must be the repo root (the dir with pyproject.toml and the
# two scripts), not just the data directory.
if [ ! -f pyproject.toml ] || [ ! -f analyze_results.py ]; then
    echo "ERROR: $PROJECT_DIR is not the rcl-market-assessment repo root" >&2
    echo "       (no pyproject.toml / analyze_results.py). Set PROJECT_DIR." >&2
    exit 1
fi

# Sync the committed code so post-processing runs against the current
# analyze_results.py / plot_results.py, not whatever the OAK clone last had.
# --ff-only never makes a merge commit and aborts cleanly if the working copy
# has diverged. results/ is gitignored, so the pull never touches generated
# data. Made non-fatal (PROJECT_DIR may not be a git checkout) and skippable:
#     sbatch --export=GIT_PULL=0 slurm/postprocess.sbatch
if [ "${GIT_PULL:-1}" != "0" ] && [ -d .git ]; then
    echo "[$(date -Iseconds)] git pull --ff-only in $PROJECT_DIR"
    git pull --ff-only || echo "[WARN] git pull failed; continuing with current code"
else
    echo "[$(date -Iseconds)] skipping git pull (GIT_PULL=0 or not a git checkout)"
fi

# Make the repo's flat modules importable for the precondition's `import
# analyze_results` below. pyproject.toml declares a [project], so uv treats the
# repo as a package and does NOT put the repo root on sys.path; on Sherlock the
# cwd is not auto-added either, so the import would fail with ModuleNotFoundError
# without this. (The two scripts themselves run as __main__ and don't need it,
# but it is harmless for them.)
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# uv on PATH (sbatch runs a non-login shell, so .bashrc may not have sourced it).
if ! command -v uv >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
fi

# Modern GCC so any source-build fallbacks (matplotlib's contourpy) can compile.
ml gcc/12.4.0 2>/dev/null || ml gcc/10.1.0 2>/dev/null || ml gcc 2>/dev/null || true

uv sync --quiet      # refresh .venv/ to match uv.lock; no-op if up to date

# -----------------------------------------------------------------------------
# PRECONDITION — prove the scripts will (a) find at least one dataset's inputs
# and (b) write their outputs under OAK, BEFORE running anything. This is the
# real safety net for this job: it resolves _find()/_analysis_dir()/_graphs_dir()
# exactly as the scripts do, prints the target dirs, and fails loudly if a local
# results/ would shadow the OAK data (sending outputs to the wrong place) or if
# no inputs exist at all. A dataset whose inputs are missing is just skipped.
# -----------------------------------------------------------------------------
echo "[$(date -Iseconds)] checking inputs + resolving output locations ..."
uv run python - <<'PY'
import sys
import analyze_results as A
import plot_results as P

OAK = A._OAK_ROOT
DATASETS = [
    ('BLP',  'blp/blp_multistart_all.csv',  'blp/blp_elasticities_detail.csv'),
    ('Nevo', 'nevo/multistart_all.csv',     'nevo/elasticities_detail.csv'),
]

found, off_oak = [], []
for name, mall, elas in DATASETS:
    try:
        csv = A._find(mall)
        A._find(elas)                      # both inputs are required per dataset
    except FileNotFoundError as e:
        print(f'  {name:4s}: inputs not found -> will be SKIPPED ({e})')
        continue
    adir, gdir = A._analysis_dir(csv), P._graphs_dir(csv)
    on_oak = adir.is_relative_to(OAK) and gdir.is_relative_to(OAK)
    tag = 'OK' if on_oak else 'NOT ON OAK'
    print(f'  {name:4s}: inputs   {csv}')
    print(f'         analysis -> {adir}   [{tag}]')
    print(f'         graphs   -> {gdir}   [{tag}]')
    found.append(name)
    if not on_oak:
        off_oak.append(name)

if not found:
    print(f"ERROR: no dataset inputs found under PROJECT_DIR/results or "
          f"{OAK / 'results'}.", file=sys.stderr)
    print("       Run blp_blp.py / nevo_blp.py first to generate the CSVs.",
          file=sys.stderr)
    sys.exit(1)
if off_oak:
    print(f"ERROR: outputs for {off_oak} would NOT land under {OAK}.",
          file=sys.stderr)
    print("       A local PROJECT_DIR/results is shadowing the OAK data.",
          file=sys.stderr)
    print("       Use a checkout whose results/ IS the OAK dir, or one with no",
          file=sys.stderr)
    print("       local results/ so the scripts fall through to OAK.",
          file=sys.stderr)
    sys.exit(1)
print(f"precondition OK: will post-process {found}")
PY

# -----------------------------------------------------------------------------
# STEP 1 / 2 — Text analysis reports -> results/<ds>/analysis/NN_*.txt
# -----------------------------------------------------------------------------
echo "[$(date -Iseconds)] STEP 1/2 analyze_results.py"
uv run python analyze_results.py

# -----------------------------------------------------------------------------
# STEP 2 / 2 — Figures -> results/<ds>/graphs/NN_*.png
# -----------------------------------------------------------------------------
echo "[$(date -Iseconds)] STEP 2/2 plot_results.py"
uv run python plot_results.py

OAK_RESULTS="/oak/stanford/groups/polinsky/blp_nevo/results"
echo "[$(date -Iseconds)] done."
echo "analysis: $OAK_RESULTS/{blp,nevo}/analysis/"
echo "graphs:   $OAK_RESULTS/{blp,nevo}/graphs/"
echo "pull both to your laptop with:"
echo "  rsync -avz asuarezg@login.sherlock.stanford.edu:'$OAK_RESULTS/' \\"
echo "             './results/' --include='*/' --include='analysis/***' \\"
echo "             --include='graphs/***' --exclude='*'"
