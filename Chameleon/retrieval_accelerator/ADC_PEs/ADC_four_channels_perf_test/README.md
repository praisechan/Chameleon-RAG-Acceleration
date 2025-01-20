# ADC_four_channels_perf_test

Send dummy LUT on-chip, thus avoiding using a DRAM channel to store LUT which downgrades the performance. For the project that use DRAM channel for LUT storage, see ADC_four_channels_functionality_test.

## Performance

Freq = 134 MHz

query num = 1000, compute iter per PE = 1000, LUT load = 256 CC, ADC pipeline depth = 131

Predicted performance: (1000 * 32 * (1000 + 256 + 131)) / (134 * 1e6) = 0.3312 sec

Real performance: 347.18 ms, same as predicted