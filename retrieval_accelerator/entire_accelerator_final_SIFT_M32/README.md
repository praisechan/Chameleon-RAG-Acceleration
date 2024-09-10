# entire_accelerator_final

This is the functionality test of entire accelerator (send query & center vectors from DRAM channels, thus performance is suboptimal). 

This project simulates the situation that query_id, cell_IDs, and cell centroids vectors are streamed in from network, and the results are streamed out to network.

Compare to entire_accelerator_with_LUT_network_simulation, this project allow user to configure double buffering for ADC computation, and fully sort the topK results before returning them.

## Performance

The double-buffering can lower the frequency significantly! Set 135MHz, real 117 MHz.


query num = 10000, compute iter per PE = 1e9/32768/8=3814, LUT load = 256 CC, ADC pipeline depth = 131

Predicted performance (without loading LUT): (10000 * 32 * (3814 + 256 + 131)) / (117 * 1e6) = 11.48 sec

Predicted performance (with loading LUT): (10000 * 32 * (256 * 32 * 4 / 64 + 3814 + 256 + 131)) / (117 * 1e6) = 12.89 sec

Real performance (12.69 sec), correct:

```
Compute Unit Utilization
Device,Compute Unit,Kernel,Global Work Size,Local Work Size,Number Of Calls,Dataflow Execution,Max Overlapping Executions,Dataflow Acceleration,Total Time (ms),Minimum Time (ms),Average Time (ms),Maximum Time (ms),Clock Frequency (MHz),
xilinx_u250_gen3x16_xdma_shell_4_1-0,vadd_1,vadd,1:1:1,1:1:1,1,Yes,1,1.000000x,12693.7,12693.7,12693.7,12693.7,117,
```