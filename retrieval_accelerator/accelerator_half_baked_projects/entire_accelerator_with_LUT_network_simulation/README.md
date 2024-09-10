# entire_accelerator_with_LUT_network_simulation

This is the functionality test of entire accelerator (send query & center vectors from DRAM channels, thus performance is suboptimal). 

This project simulates the situation that query_id, cell_IDs, and cell centroids vectors are streamed in from network, and the results are streamed out to network.

Compare to entire_accelerator_network_simulation, this project add ADC computation unit instead of taking LUTs from CPU. The ADC unit does not have double-buffering.

## Performance
