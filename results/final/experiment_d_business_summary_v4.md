# Experiment D — business summary

Walk-forward on S&P 500 (90 tickers, 2012-2025) and MOEX (28 tickers, 2014-2025).
3-year training window, monthly rebalance, transaction costs 10 bps (SP) / 30 bps (MOEX).

## SP500

| Strategy                    |   CAGR % |   Sharpe |   Sortino |   Max DD % |   Ann. Turnover % |   Ann. Costs bps |   Net excess vs 1/N (bps) |
|:----------------------------|---------:|---------:|----------:|-----------:|------------------:|-----------------:|--------------------------:|
| 1/N Equal weight            |   15.993 |    0.948 |     0.196 |    -31.958 |             4.719 |            0.472 |                     0     |
| Continuous MVO              |   16.414 |    0.877 |     0.247 |    -26.419 |           210.741 |           21.074 |                    42.101 |
| Continuous + Top-K rounding |   16.396 |    0.876 |     0.247 |    -26.419 |           210.799 |           21.08  |                    40.292 |
| SCIP MIQP (discrete)        |   16.381 |    0.875 |     0.247 |    -26.436 |           210.96  |           21.096 |                    38.783 |
| Neal SA (quantum-inspired)  |   14.398 |    0.836 |     0.215 |    -28.219 |           267.601 |           26.76  |                  -159.494 |
| Tabu SA (quantum-inspired)  |   16.64  |    0.885 |     0.249 |    -26.42  |           210.423 |           21.042 |                    64.732 |

## MOEX

| Strategy                    |   CAGR % |   Sharpe |   Sortino |   Max DD % |   Ann. Turnover % |   Ann. Costs bps |   Net excess vs 1/N (bps) |
|:----------------------------|---------:|---------:|----------:|-----------:|------------------:|-----------------:|--------------------------:|
| 1/N Equal weight            |    5.678 |    0.374 |     0.094 |    -41.548 |             5.951 |            1.785 |                     0     |
| Continuous MVO              |    4.572 |    0.331 |     0.098 |    -39.759 |           150.454 |           45.136 |                  -110.542 |
| Continuous + Top-K rounding |    4.678 |    0.336 |     0.099 |    -39.759 |           150.788 |           45.236 |                   -99.962 |
| SCIP MIQP (discrete)        |    4.683 |    0.336 |     0.099 |    -39.774 |           150.758 |           45.227 |                   -99.495 |
| Neal SA (quantum-inspired)  |    4.274 |    0.312 |     0.094 |    -39.662 |           155.844 |           46.753 |                  -140.354 |
| Tabu SA (quantum-inspired)  |    4.637 |    0.334 |     0.098 |    -39.774 |           150.507 |           45.152 |                  -104.076 |
