# Paper build

```bash
# 1. Run experiments to produce runs/sweeps/summary.json
python scripts/run_sweeps.py --dataset toy --n 30

# 2. Generate paper/tables/*.tex from the summary
python scripts/collate_results.py --in runs/sweeps/summary.json --out paper/tables

# 3. Compile (requires TeXLive)
cd paper && latexmk -pdf main.tex
```

`refs.bib` is seeded with the cited systems. Several entries are placeholders
(`Anonymous` / approximate arXiv IDs) and should be replaced with the exact
citations before submission.
