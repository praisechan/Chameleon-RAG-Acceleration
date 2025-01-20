# ADC_PE_single_channel_double_buffer

Unfixed iteration performance test of ADC PE. Without proper control logic yet.

Predicted performance:

For M = 32, ADC L = 131 cycles, LUT init = 256 cycles (hidden)

`query num = 10000; nprobe = 32; compute_iter_per_PE = 1000;` -> 10000 * 32 * (131 + 1000) cycles ; 180 MHz -> 10000 * 32 * (131 + 1000) / (180 * 1e6) = 2.01 sec

Real performance (summary.csv):

2036.1 ms (same)