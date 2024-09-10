# entire_accelerator_without_LUT_network_simulation

This is the functionality test of entire accelerator without LUT construction PE (send LUT from one of the DRAM channels, thus performance is suboptimal), for perf test, see entire_accelerator_perf_test. 

This project simulates the situation that query_id, cell_IDs, and LUTs are streamed in from network, and the results are streamed out to network.

## Performance

136 MHz 

query num = 1000, compute iter per PE = 1000, LUT load = 256 CC, ADC pipeline depth = 131

Predicted performance (without loading LUT): (1000 * 32 * (1000 + 256 + 131)) / (136 * 1e6) = 0.326 sec

Predicted performance (with loading LUT): (1000 * 32 * (256 * 32 * 4 / 64 + 1000 + 256 + 131)) / (136 * 1e6) = 0.447 sec

Real performance: 398.95 ms, even better than the predicted worst case performance... Probably in this case, stream LUT from DRAM overlaps with PE LUT init, thus the time consumption is reduced!