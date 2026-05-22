.PHONY: install test sweep tables figures paper clean all

PY ?= python
DATASET ?= hotpotqa
N ?= 120
TOPK ?= 3
SWEEP_DIR ?= runs/sweeps_hotpot
GEN ?= extractive

install:
	$(PY) -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]" && pip install datasets sentence-transformers faiss-cpu matplotlib

test:
	$(PY) -m pytest tests/ -q

sweep:
	RAGCACHE_GENERATOR=$(GEN) $(PY) scripts/run_sweeps.py \
		--dataset $(DATASET) --n $(N) --top-k $(TOPK) --out-dir $(SWEEP_DIR)

tables: $(SWEEP_DIR)/summary.json
	$(PY) scripts/collate_results.py --in $(SWEEP_DIR)/summary.json --out paper/tables
	$(PY) scripts/results_text.py --in $(SWEEP_DIR)/summary.json --out paper/tables/results_text.tex

figures: $(SWEEP_DIR)/summary.json
	$(PY) scripts/make_figures.py --in $(SWEEP_DIR)/summary.json --out-dir paper/figures

paper: tables figures
	cd paper && latexmk -pdf main.tex

all: sweep tables figures paper

clean:
	rm -rf runs/ paper/tables/*.tex paper/figures/*.pdf paper/main.pdf paper/*.aux paper/*.log paper/*.bbl paper/*.blg paper/*.fls paper/*.fdb_latexmk paper/*.out

$(SWEEP_DIR)/summary.json:
	@echo "Run 'make sweep' first to generate $@"
	@exit 1
