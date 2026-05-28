# Experiment D — business summary

Walk-forward on S&P 500 (90 tickers, 2012-2025) and MOEX (28 tickers, 2014-2025).
3-year training window, monthly rebalance, transaction costs 10 bps (SP) / 30 bps (MOEX).

## SP500

| Strategy                    |   CAGR % |   Sharpe |   Sortino |   Max DD % |   Ann. Turnover % |   Ann. Costs bps |   Net excess vs 1/N (bps) |
|:----------------------------|---------:|---------:|----------:|-----------:|------------------:|-----------------:|--------------------------:|
| 1/N Equal weight            |   15.993 |    0.948 |     0.196 |    -31.958 |             4.719 |            0.472 |                     0     |
| Continuous MVO              |   23.927 |    0.837 |     0.299 |    -43.595 |           292.884 |           29.288 |                   793.41  |
| Continuous + Top-K rounding |   23.927 |    0.837 |     0.299 |    -43.595 |           292.885 |           29.288 |                   793.403 |
| SCIP MIQP (discrete)        |   23.931 |    0.837 |     0.299 |    -43.595 |           292.991 |           29.299 |                   793.79  |
| Neal SA (quantum-inspired)  |   24.834 |    0.861 |     0.324 |    -43.899 |           368.115 |           36.811 |                   884.054 |

## MOEX

| Strategy                    |   CAGR % |   Sharpe |   Sortino |   Max DD % |   Ann. Turnover % |   Ann. Costs bps |   Net excess vs 1/N (bps) |
|:----------------------------|---------:|---------:|----------:|-----------:|------------------:|-----------------:|--------------------------:|
| 1/N Equal weight            |   92.473 |    0.355 |    22.52  |    -41.548 |             5.951 |            1.785 |                      0    |
| Continuous MVO              |   -4.479 |    0.045 |     0.018 |    -71.916 |           185.572 |           55.672 |                  -9695.11 |
| Continuous + Top-K rounding |   -4.484 |    0.045 |     0.018 |    -71.916 |           185.63  |           55.689 |                  -9695.7  |
| SCIP MIQP (discrete)        |   -4.477 |    0.045 |     0.018 |    -71.904 |           185.562 |           55.669 |                  -9694.96 |
| Neal SA (quantum-inspired)  |   -4.502 |    0.044 |     0.017 |    -72.207 |           197.373 |           59.212 |                  -9697.48 |
