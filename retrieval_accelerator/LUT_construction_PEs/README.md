# LUT Construction PE

For vectors of small dimensions, e.g., D <= 128, we can use many sub-PEs to construct distance LUT, and output 1 row of LUT per cycle. (LUT_construction_PE_D128_M32)

However, if the dimension of vectors is higher, e.g., >= 256 (and M is higher). The sub-PEs consume quite some resources and will lead to P&R problems. I tried a couple of solutions.

(1) Still keep 1 sub-PE per M, but let the pipeline II to be more than one to reduce resource consumption. However, this solution does not work well. (for succeeding P&R) (LUT_construction_PE_D384_M64_slow_pipeline)

(2) Use less sub-PEs, e.g,. 16 PE for M=64 -> 4 cycles per row of LUT. This method work efficiently in succeeding P&R. (LUT_construction_PE_D384_M64_v3_fixed_PE_num)