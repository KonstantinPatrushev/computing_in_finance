.PHONY: help env env-update install data data-sp500 data-moex test lint clean reproduce reproduce-quantum

CONDA_ENV := cif
PY := python

help:
	@echo "Targets:"
	@echo "  env                  Create conda environment from environment.yml"
	@echo "  env-update           Update existing conda environment"
	@echo "  install              Editable pip install of the package"
	@echo "  data                 Download S&P500 + MOEX data"
	@echo "  data-sp500           Download only S&P500 data"
	@echo "  data-moex            Download only MOEX data"
	@echo "  test                 Run pytest"
	@echo "  lint                 Run ruff linter"
	@echo "  reproduce            Reproduce classical + simulator experiments"
	@echo "  reproduce-quantum    Reproduce real quantum experiments (uses quotas)"
	@echo "  clean                Remove caches and build artifacts"

env:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

install:
	$(PY) -m pip install -e .

data: data-sp500 data-moex

data-sp500:
	$(PY) -m cif.scripts.download_data --universe sp500

data-moex:
	$(PY) -m cif.scripts.download_data --universe moex

test:
	$(PY) -m pytest tests/

lint:
	$(PY) -m ruff check src/ tests/

reproduce: reproduce-a reproduce-d
	@echo "Classical + quantum-inspired reproduction complete. See results/final/."

reproduce-a:
	$(PY) scripts/experiment_a_scalability.py \
	    --N 20 30 50 75 100 150 200 \
	    --seeds 42 123 7 \
	    --time-limit 300 \
	    --source synthetic
	$(PY) scripts/aggregate_experiment_a.py

reproduce-b:
	$(PY) scripts/experiment_b_time_budget.py \
	    --N 50 100 150 200 \
	    --budgets 0.5 1.0 3.0 10.0 \
	    --seeds 42 123 7

reproduce-d:
	$(PY) scripts/experiment_d_walkforward.py
	$(PY) scripts/aggregate_experiment_d.py --tag v1
	$(PY) scripts/experiment_d_walkforward_v2.py --suffix v2
	$(PY) scripts/aggregate_experiment_d.py --tag v2
	$(PY) scripts/experiment_d_walkforward_v2.py --suffix v3
	$(PY) scripts/aggregate_experiment_d.py --tag v3

report:
	cd notebooks && jupyter nbconvert --to notebook --execute \
	    99_final_report.ipynb --output 99_final_report.ipynb \
	    --ExecutePreprocessor.timeout=600

reproduce-quantum:
	@echo "Experiment C (QAOA GPU) must run on HSE HPC — see REPRODUCIBILITY.md"
	@echo "This target is a placeholder; do not run locally unless you have a CUDA GPU."

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
