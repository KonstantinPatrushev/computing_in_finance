# Experiment D — business summary

Walk-forward on S&P 500 (90 tickers, 2012-2025) and MOEX (28 tickers, 2014-2025).
3-year training window, monthly rebalance, transaction costs 10 bps (SP) / 30 bps (MOEX).

## SP500

| Strategy                    |   CAGR % |   Sharpe |   Sortino |   Max DD % |   Ann. Turnover % |   Ann. Costs bps |   Net excess vs 1/N (bps) |
|:----------------------------|---------:|---------:|----------:|-----------:|------------------:|-----------------:|--------------------------:|
| 1/N Equal weight            |   15.993 |    0.948 |     0.196 |    -31.958 |             4.719 |            0.472 |                     0     |
| Continuous MVO              |   16.382 |    0.874 |     0.247 |    -26.419 |           211.576 |           21.158 |                    38.872 |
| Continuous + Top-K rounding |   16.364 |    0.874 |     0.247 |    -26.419 |           211.635 |           21.163 |                    37.061 |
| SCIP MIQP (discrete)        |   16.356 |    0.873 |     0.247 |    -26.437 |           211.717 |           21.172 |                    36.248 |
| Neal SA (quantum-inspired)  |   11.669 |    0.723 |     0.179 |    -25.883 |           872.906 |           87.291 |                  -432.395 |

## MOEX

| Strategy                    |   CAGR % |   Sharpe |   Sortino |   Max DD % |   Ann. Turnover % |   Ann. Costs bps |   Net excess vs 1/N (bps) |
|:----------------------------|---------:|---------:|----------:|-----------:|------------------:|-----------------:|--------------------------:|
| 1/N Equal weight            |   92.473 |    0.355 |    22.52  |    -41.548 |             5.951 |            1.785 |                      0    |
| Continuous MVO              |    3.797 |    0.283 |     0.081 |    -39.759 |           146.31  |           43.893 |                  -8867.56 |
| Continuous + Top-K rounding |    4.402 |    0.312 |     0.091 |    -39.759 |           153.792 |           46.138 |                  -8807.05 |
| SCIP MIQP (discrete)        |    4.409 |    0.312 |     0.091 |    -39.774 |           153.726 |           46.118 |                  -8806.32 |
| Neal SA (quantum-inspired)  |    2.236 |    0.201 |     0.061 |    -38.876 |           549.628 |          164.888 |                  -9023.62 |
