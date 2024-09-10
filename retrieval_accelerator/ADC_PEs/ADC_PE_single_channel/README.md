# ADC_PE_single_channel

Unfixed iteration performance test of ADC PE. Without proper control logic yet.

Predicted performance:

For M = 32, ADC L = 131 cycles, LUT init = 256 cycles

`query num = 10000; nprobe = 32; compute_iter_per_PE = 1000;` -> 10000 * 32 * (131 + 256 + 1000) cycles ; 200 MHz -> 10000 * 32 * (131 + 256 + 1000) / (200 * 1e6) = 2.2192 sec

Real performance (summary.csv):

2278.61 ms (Same)

## Implication

The compute pipeline latency \& LUT instantiation is a significant overhead (40\% Given 1000 iterations)

Suppose an FPGA contain nlist = 64K clusters. 

For 8-byte PQ codes -> 4 channel x 8 PE per channel = 32 PEs

Compute iterations per cell = NVEC / 64K / 32 = NVEC / 2M 

NVEC = 2B -> 1K iterations



