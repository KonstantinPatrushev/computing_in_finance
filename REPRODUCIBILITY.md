# Reproducibility guide

Точные команды для воспроизведения всех четырёх экспериментов (A, B, C, D) и построения финального отчёта.

## Предусловия

- Linux или macOS
- Anaconda/Miniconda
- ~5 GB свободного места (conda env + data + results)
- Интернет для фазы загрузки данных

## 0. Setup

```bash
git clone <repo-url>
cd computing_in_finance

# Создать conda env (займёт ~5-15 минут)
make env
conda activate cif

# Зарегистрировать kernel для jupyter (для исполнения notebooks)
python -m ipykernel install --user --name cif --display-name "cif"
```

Если нужен GPU-симулятор для Experiment C — на HPC-узле с NVIDIA GPU:
```bash
pip install qiskit-aer-gpu cuquantum-python
```

## 1. Загрузка данных (только локально — требует интернет)

```bash
python -m cif.scripts.download_data --universe both --refresh-universe
```

Скачивает:
- `data/raw/universe_sp500.json`, `data/raw/universe_moex.json` — снепшоты universe
- `data/raw/{sp500,moex}_prices_raw.parquet` — сырые цены
- `data/processed/{sp500,moex}_prices.parquet` — очищенный price panel
- `data/processed/{sp500,moex}_returns.parquet` — log returns
- `data/processed/{sp500,moex}_provenance.json` — SHA256 + cleaning report

Ожидаемая форма:
- SP500: 90 тикеров × 3427 дней (2012-05-15 … 2025-12-30)
- MOEX: 28 тикеров × 2895 дней (2014-06-09 … 2025-12-30)

## 2. Unit tests (~1 сек)

```bash
pytest tests/ -v
```

16 тестов, все должны быть зелёными. Покрывают encoding round-trip, QUBO energy vs objective, классические solver'ы на аналитических примерах N=2 и sanity check brute-force vs SCIP.

## 3. Эксперименты

### Experiment A — scalability (дискретная оптимизация на синтетике)

```bash
python scripts/experiment_a_scalability.py \
    --N 20 30 50 75 100 150 200 \
    --seeds 42 123 7 \
    --time-limit 300 \
    --source synthetic

python scripts/aggregate_experiment_a.py
```

Ожидаемое время: ~20-30 минут. Артефакты:
- `results/experiment_a.jsonl`
- `results/final/experiment_a_summary.csv`
- `results/final/figures/exp_a_scalability.png` (log-log wall time vs N)
- `results/final/figures/exp_a_quality.png` (gap % vs N)

### Experiment B — quality at fixed time budget

```bash
python scripts/experiment_b_time_budget.py \
    --N 50 100 150 200 \
    --budgets 0.5 1.0 3.0 10.0 \
    --seeds 42 123 7
```

Ожидаемое время: ~30-60 минут (много комбинаций).

### Experiment C — QAOA на симуляторе

**Локальный запуск (только CPU, N ≤ 14):**
```bash
jupyter nbconvert --to notebook --execute \
    notebooks/hpc_experiment_c_qaoa_gpu.ipynb \
    --output hpc_experiment_c_qaoa_gpu_cpu_only.ipynb
```

**HPC-запуск (GPU, N до 24):**
1. На локальной машине: `scp -r src data notebooks pyproject.toml environment.yml user@hpc:~/cif/`
2. На HPC: `conda env create -f environment.yml && conda activate cif && pip install qiskit-aer-gpu cuquantum-python`
3. SSH + jupyter tunnel, открыть `notebooks/hpc_experiment_c_qaoa_gpu.ipynb`, выполнить
4. `scp user@hpc:~/cif/results/experiment_c_qaoa.csv results/`

### Experiment D — walk-forward backtest

**v1 (мягкие ограничения, демонстрирует MVO-концентрацию):**
```bash
python scripts/experiment_d_walkforward.py
python scripts/aggregate_experiment_d.py --tag v1
```

**v2 (жёсткие ограничения `w_max=0.10, K=15` для SP500):**
```bash
python scripts/experiment_d_walkforward_v2.py --suffix v2
python scripts/aggregate_experiment_d.py --tag v2
```

**v3 (persistent neal с warm-start — рекомендуемый финальный вариант):**
```bash
python scripts/experiment_d_walkforward_v2.py --suffix v3
python scripts/aggregate_experiment_d.py --tag v3
```

Ожидаемое время: ~5 минут за прогон. Артефакты:
- `results/experiment_d_{universe}_{tag}_summary.csv` (5 стратегий × 2 универса)
- `results/experiment_d_{universe}_{tag}_folds.jsonl` (per-fold детали)
- `results/final/experiment_d_business_table_{tag}.csv`
- `results/final/figures/exp_d_{universe}_{tag}_equity.png`

## 4. Финальный отчёт

```bash
jupyter nbconvert --to notebook --execute \
    notebooks/99_final_report.ipynb --output 99_final_report.ipynb
```

## Фиксированные версии

См. `environment.yml` и `pyproject.toml`. Ключевые (на момент 2026-04-15):
- `cvxpy 1.6.7`, `pulp 2.8`, `pyscipopt 6.1.0`
- `dimod 0.12.21`, `dwave-ocean-sdk 9.3.0`
- `qiskit 2.3.1`, `qiskit-aer 0.17.2`, `qiskit-ibm-runtime 0.46.1`, `qiskit-optimization 0.7.0`
- `yfinance 1.2.2`, `apimoex 1.4.0`
- `numpy 2.1.3`, `pandas 2.2.3`, `scipy 1.x`, `scikit-learn 1.5`

## Known quirks

- **IDE hints** в VSCode/Cursor могут показывать "package not installed" если не выбран `cif` interpreter. Env реально установлен и импорты работают.
- **Wikipedia блокирует** default urllib User-Agent, поэтому `universe.py` явно ставит UA header.
- **MOEX ISS analytics endpoint** пагинирует по 20 строк — использует `limit=200` для полного списка IMOEX.
- **ECOS_BB** — встроенный в cvxpy branch-and-bound. Иногда faster чем SCIP на малых N, но scales хуже.
- **SCIP timing variance** на N≥150 огромный (std >> mean) — это особенность его адаптивных эвристик на случайных инстансах.
- **neal turnover в walk-forward** без persistent warm-start превышает 500% (SA находит разные subsets каждый фолд). v3 с warm-start это исправляет.
