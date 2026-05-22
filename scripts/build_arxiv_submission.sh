#!/usr/bin/env bash
# Build a clean arXiv submission tarball from paper/.
#
# arXiv expects: .tex sources + .bib (or .bbl) + figures, packaged as tar.gz.
# No PDFs of the final paper, no .aux/.log/.out, no hidden files.
#
# Usage: bash scripts/build_arxiv_submission.sh
# Output: arxiv_submission.tar.gz (in repo root)
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="arxiv_submission.tar.gz"
STAGE="$(mktemp -d)/cache_llm"
mkdir -p "$STAGE"

# Copy the paper tree, only the files arXiv needs.
rsync -a --prune-empty-dirs \
  --include='*/' \
  --include='main.tex' --include='refs.bib' \
  --include='tables_hotpotqa/*.tex' --include='tables_mtrag/*.tex' \
  --include='figures/*.png' --include='figures/*.pdf' \
  --include='figures_hotpotqa/*.pdf' --include='figures_mtrag/*.pdf' \
  --exclude='*' \
  paper/ "$STAGE/"

# Include a pre-built .bbl so arXiv doesn't need to re-run BibTeX.
# (Only present if you compiled main.tex locally and produced paper/main.bbl.)
if [ -f paper/main.bbl ]; then
  cp paper/main.bbl "$STAGE/main.bbl"
  echo "[ok] included paper/main.bbl"
else
  echo "[warn] paper/main.bbl not found. arXiv will run BibTeX from refs.bib."
  echo "       To pre-build: compile main.tex locally (pdflatex + bibtex + pdflatex + pdflatex)"
fi

# Sanity scan for forbidden patterns.
echo "[scan] checking for arXiv blockers..."
! grep -rn -- '\\today' "$STAGE" || { echo "ERROR: \\today found"; exit 1; }
! find "$STAGE" -name '.*' -print -quit | grep -q . || { echo "ERROR: hidden files present"; exit 1; }

tar -C "$(dirname "$STAGE")" -czf "$OUT" "$(basename "$STAGE")"
echo "[done] $OUT  ($(du -h "$OUT" | cut -f1))"
echo "Upload $OUT at https://arxiv.org/submit"
