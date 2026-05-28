# computing_in_finance

Исследование возможностей применения гибридных квантово-классических алгоритмов для решения дискретной задачи Марковица.

Сравнение ведётся на двух универсах (S&P 500 top-100 + MOEX IMOEX constituents) для шести солверов:

- **Brute force** — exact ground truth для N ≤ 12.
- **CVXPY continuous MVO** — непрерывный референс.
- **PuLP + CBC MIQP** — certified discrete ground truth с cardinality.
- **D-Wave neal SA** — классический симулятор annealing (quantum-inspired).
- **D-Wave Leap Hybrid CQM** — реальный квантовый гибридный сервис.
- **Qiskit QAOA** — gate-based, на Aer и IBM Quantum Heron.

## Quick start

```bash
# Создать conda env
make env
conda activate cif

# Скачать данные локально (требует интернет)
python -m cif.scripts.download_data --universe both

# Прогнать классику + симулятор локально
make reproduce

# Прогнать реальные квантовые эксперименты (квоты!)
cp .env.example .env  # вставить токены IBM_QUANTUM_TOKEN, DWAVE_API_TOKEN
make reproduce-quantum
```

## Архитектура: локальная машина vs суперкомпьютер ВШЭ

Compute-узлы кластера ВШЭ **не имеют исходящего интернета**, поэтому проект жёстко разделён на две части:

### Локально (требует интернет)
- `python -m cif.scripts.download_data` — скачивание yfinance + MOEX данных в parquet
- DWave Leap Hybrid CQM, DWave QPU direct
- IBM Quantum Runtime (QAOA на Heron r2/r3)

### На HPC (через scp + ssh + jupyter tunnel)
- Brute force большого N
- PuLP + CBC MIQP solving
- DWave neal SA (классический)
- Qiskit Aer симулятор QAOA
- Walk-forward backtest со множеством фолдов и сидов
- λ-ablation grids
- Статистические тесты

### Workflow

1. **Локально:** `make env`, `python -m cif.scripts.download_data --universe both`
2. **Перенос на HPC:** `scp` → загрузить `data/processed/*.parquet`, исходники `src/cif`, `pyproject.toml`, `environment.yml`, `notebooks/hpc_*.ipynb`
3. **На HPC:** `conda env create -f environment.yml && conda activate cif`, ssh + jupyter notebook + tunnel, открыть `notebooks/hpc_*.ipynb`, выполнить все ячейки
4. **Обратно:** `scp` результирующие JSONL/parquet с HPC локально
5. **Локально:** агрегация, графики, quantum API jobs (DWave Leap, IBM Quantum)
6. **Локально:** сборка финального отчёта `notebooks/99_final_report.ipynb`

## Структура

```
src/cif/
├── data/         # universe, providers (yfinance, MOEX), cleaning, statistics
├── classical/    # cvxpy continuous, brute force, pulp+CBC MIQP
├── qubo/         # encoding, BQM/CQM builders, penalties, validation
├── quantum/      # neal, dwave hybrid, qiskit QAOA, decoder
├── experiments/  # runner, grid, JSONL registry
├── backtest/     # walk-forward engine, transaction costs
├── metrics/      # objective, feasibility, quality (approximation ratio, TTS)
├── viz/          # frontier, convergence, comparison plots
└── scripts/      # CLI entry points (download_data, run_experiment, aggregate)
```

## Зависимости

См. `pyproject.toml` и `environment.yml`. Ключевые: `cvxpy`, `pulp`, `dimod`, `dwave-ocean-sdk`, `qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`, `qiskit-optimization`, `yfinance`, `apimoex`, `scikit-learn`.
