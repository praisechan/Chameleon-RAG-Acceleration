# entire_accelerator_functionality_test

This is the functionality test of entire accelerator (send LUT from one of the DRAM channels, thus performance is suboptimal), for perf test, see entire_accelerator_perf_test. 

## Performance

136 MHz 

query num = 1000, compute iter per PE = 1000, LUT load = 256 CC, ADC pipeline depth = 131

Predicted performance (without loading LUT): (1000 * 32 * (1000 + 256 + 131)) / (136 * 1e6) = 0.326 sec

Predicted performance (with loading LUT): (1000 * 32 * (256 * 32 * 4 / 64 + 1000 + 256 + 131)) / (136 * 1e6) = 0.447 sec

Real performance: 441.418, same as predicted (with loading LUT)