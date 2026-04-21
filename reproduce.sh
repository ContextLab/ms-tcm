#!/usr/bin/env bash
# reproduce.sh — end-to-end reproducer for the MS-TCM paper.
#
# Performs, in order (each step idempotent; skipped if its output is cached):
#   1. Verify Python 3.11 is available (or install via Homebrew on macOS /
#      apt on Debian-ish Linux / instruct the user on Windows).
#   2. Create / refresh a .venv virtualenv.
#   3. Install the ms_tcm package (editable) with extras + analysis deps.
#   4. Initialize the CDL-bibliography submodule.
#   5. Validate the bundled FRFR-category dataset.
#   6. Compute sentence-transformer embeddings for every word.
#   7. Fit MS-TCM on FRFR-category (caches to data/processed/fits/mstcm/).
#   8. Fit standard TCM (w_S=0) (caches to data/processed/fits/tcm/).
#   9. Regenerate every figure into paper/figs/source/.
#  10. Compile paper/main.pdf.
#
# Flags:
#   --force-rerun             ignore every cache and rerun everything
#   --skip-fits               skip steps 7-8 (useful if fits were produced externally)
#   --skip-paper              skip step 10 (LaTeX)
#   -h | --help               show this help

set -euo pipefail

# ---- Argument parsing ------------------------------------------------------
FORCE_RERUN=0
SKIP_FITS=0
SKIP_PAPER=0
for arg in "$@"; do
    case "$arg" in
        --force-rerun) FORCE_RERUN=1 ;;
        --skip-fits) SKIP_FITS=1 ;;
        --skip-paper) SKIP_PAPER=1 ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

FORCE_FLAG=""
if [[ $FORCE_RERUN -eq 1 ]]; then
    FORCE_FLAG="--force-rerun"
fi

# ---- 1. Python 3.11 --------------------------------------------------------
echo "=== Step 1: locate Python 3.11 ==="
PYTHON_BIN=""
for candidate in python3.11 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 /usr/bin/python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo "Python 3.11 not found. Attempting to install..."
    case "$(uname -s)" in
        Darwin)
            if ! command -v brew >/dev/null 2>&1; then
                echo "Homebrew is not installed. Install from https://brew.sh and re-run." >&2
                exit 1
            fi
            brew install python@3.11
            PYTHON_BIN="$(brew --prefix)/bin/python3.11"
            ;;
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                sudo apt-get update
                sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
                PYTHON_BIN=python3.11
            else
                echo "Install Python 3.11 manually for your distribution and re-run." >&2
                exit 1
            fi
            ;;
        *)
            echo "Install Python 3.11 manually for your platform and re-run." >&2
            exit 1
            ;;
    esac
fi
echo "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

# ---- 2. Virtualenv ---------------------------------------------------------
echo "=== Step 2: virtualenv ==="
if [[ ! -d .venv ]] || [[ $FORCE_RERUN -eq 1 ]]; then
    rm -rf .venv
    "$PYTHON_BIN" -m venv .venv
    echo "Created .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- 3. Install dependencies ----------------------------------------------
echo "=== Step 3: install dependencies ==="
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e ".[dev]"
# Analysis + paper deps not in pyproject (kept optional to keep the core small).
python -m pip install --quiet matplotlib sentence-transformers

# ---- 4. Submodule ----------------------------------------------------------
echo "=== Step 4: initialize paper/CDL-bibliography submodule ==="
git submodule update --init --recursive paper/CDL-bibliography || true

# ---- 5. Validate dataset ---------------------------------------------------
echo "=== Step 5: validate bundled FRFR-category dataset ==="
ms-tcm validate data/raw/frfr_category

# ---- 6. Embeddings ---------------------------------------------------------
echo "=== Step 6: compute word embeddings ==="
python code/scripts/compute_embeddings.py $FORCE_FLAG

# ---- 7 & 8. Fits -----------------------------------------------------------
if [[ $SKIP_FITS -eq 0 ]]; then
    echo "=== Step 7: fit MS-TCM (can take ~45 min on a laptop) ==="
    if [[ ! -f data/processed/fits/mstcm/fit_summary.json ]] || [[ $FORCE_RERUN -eq 1 ]]; then
        ms-tcm fit data/raw/frfr_category \
            --out data/processed/fits/mstcm \
            --seed 42 --n-bootstraps 5 --n-restarts 2 --force
    else
        echo "Cache hit: data/processed/fits/mstcm/fit_summary.json"
    fi

    echo "=== Step 8: fit standard TCM baseline ==="
    if [[ ! -f data/processed/fits/tcm/fit_summary.json ]] || [[ $FORCE_RERUN -eq 1 ]]; then
        ms-tcm fit data/raw/frfr_category \
            --out data/processed/fits/tcm \
            --seed 42 --n-bootstraps 5 --n-restarts 2 --standard-tcm --force
    else
        echo "Cache hit: data/processed/fits/tcm/fit_summary.json"
    fi
else
    echo "=== Steps 7-8: skipped (--skip-fits) ==="
fi

# ---- 9. Figures ------------------------------------------------------------
echo "=== Step 9: generate figures ==="
python code/figures/make_fig_model.py $FORCE_FLAG
python code/figures/make_fig_experiment.py $FORCE_FLAG
python code/figures/make_fig_analyses.py $FORCE_FLAG
python code/figures/make_fig_clustering.py $FORCE_FLAG
python code/figures/make_param_table.py $FORCE_FLAG

# ---- 10. Paper -------------------------------------------------------------
if [[ $SKIP_PAPER -eq 0 ]]; then
    echo "=== Step 10: compile paper ==="
    if command -v pdflatex >/dev/null 2>&1; then
        (cd paper && ./compile.sh)
    else
        echo "pdflatex not found; skipping paper compilation."
        echo "Install a TeX distribution (e.g. MacTeX, TeX Live) and rerun to build the PDF."
    fi
else
    echo "=== Step 10: skipped (--skip-paper) ==="
fi

echo "=== Done. ==="
