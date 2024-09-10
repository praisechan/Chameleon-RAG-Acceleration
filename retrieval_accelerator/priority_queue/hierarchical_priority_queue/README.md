# Probablistic Hierarchical Priority Queue

Hierarchical Priority Queue of unfixed iterations.

Performance verified, correctness not verified yet (but should be fine).

Predicted Performance: 

`query_num = 10000` `iter_num_per_query_per_ADC_PE = 10000`, thus need 1e4 * 1e4 = 1e8 cycles; set freq = 160 MHz -> should consume 1e8 / (160 * 1e6) = 0.625 s

Real Performance (from summary.csv): 

635.22 ms