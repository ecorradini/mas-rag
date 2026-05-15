PY ?= python3
PIP ?= $(PY) -m pip
SRC := src/rag_epidemic
TESTS := tests

.PHONY: install install-ablit data test lint pilot full figures rescore \
        repro ablations attack-intensity cross-domain mixed-encoder \
        ablit-qwen lock clean

install:
	$(PIP) install -e .

install-ablit:
	$(PIP) install -e .[abliteration,dev]

data:
	$(PY) -m rag_epidemic.cli.download_data --all

test:
	$(PY) -m pytest $(TESTS) -q

lint:
	$(PY) -m ruff check $(SRC) $(TESTS)

pilot:
	$(PY) -m rag_epidemic.cli.reproduce_all --mode pilot

# Token-capped reproducible run; rerun across days until done.
full:
	$(PY) -m rag_epidemic.cli.reproduce_all --mode full --daily-token-cap 9500000

# Minimal grid for fast reviewers: 5 seeds, N_A=6, medium, T=200.
repro:
	$(PY) -m rag_epidemic.cli.reproduce_all --mode full \
	  --seeds 0 1 2 3 4 --N_A 6 --difficulties medium --T 200 \
	  --daily-token-cap 9500000

# EVP ablations (J11): 3 seeds, N_A=6, medium, T=40, 7 component toggles.
ablations:
	$(PY) -m rag_epidemic.cli.run_ablations --seeds 0 1 2 --T 40

# Patient-Zero attack-intensity sweep (J12).
attack-intensity:
	$(PY) -m rag_epidemic.cli.run_attack_intensity --seeds 0 1 2 --T 40

# Cross-domain transfer probe (J15).
cross-domain:
	$(PY) -m rag_epidemic.cli.run_cross_domain --seeds 0 1 2 --T 40

# Mixed-encoder probe (J16).
mixed-encoder:
	$(PY) -m rag_epidemic.cli.run_mixed_encoder --seeds 0 1 2 --T 40

# Qwen 2.5-3B abliteration (J14). Local only; zero OpenAI cost.
ablit-qwen:
	$(PY) -m rag_epidemic.cli.ablit_cli \
	  --model-id Qwen/Qwen2.5-3B-Instruct --evaluate

figures:
	$(PY) -m rag_epidemic.analysis.make_figures --rescore

rescore:
	$(PY) -m rag_epidemic.analysis.make_figures --rescore --figs-only

lock:
	$(PIP) install pip-tools >/dev/null
	$(PY) -m piptools compile pyproject.toml -o requirements.lock.txt

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
