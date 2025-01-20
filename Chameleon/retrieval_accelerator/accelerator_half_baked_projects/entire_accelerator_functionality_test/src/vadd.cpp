#include "ADC.hpp"
#include "constants.hpp"
#include "DRAM_utils.hpp"
#include "helpers.hpp"
#include "hierarchical_priority_queue.hpp"
#include "types.hpp"

extern "C" {

void vadd(  
    // in init
    int query_num, 
    int nlist,
    int nprobe,
    int* nlist_init, // which consists of the following three stuffs:
                    // int* nlist_PQ_codes_start_addr,
                    // int* nlist_vec_ID_start_addr,
                    // int* nlist_num_vecs,

    // in runtime (should from network)
    int* cell_ID_DRAM, // query_num * nprobe
    ap_uint<512>* LUT_DRAM, // query_num * nprobe * 256 * M

    // in runtime (should from DRAM)
    const ap_uint<512>* PQ_codes_DRAM_0,
    const ap_uint<512>* PQ_codes_DRAM_1,
    const ap_uint<512>* PQ_codes_DRAM_2,
    const ap_uint<512>* PQ_codes_DRAM_3,
    ap_uint<64>* vec_ID_DRAM_0,
    ap_uint<64>* vec_ID_DRAM_1,
    ap_uint<64>* vec_ID_DRAM_2,
    ap_uint<64>* vec_ID_DRAM_3,

    // out
    ap_uint<512>* out_DRAM)
{
// Share the same AXI interface with several control signals (but they are not allowed in same dataflow)
//    https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior

// in init
#pragma HLS INTERFACE m_axi port=nlist_init  offset=slave bundle=gmem_control

// in runtime (should from network)
#pragma HLS INTERFACE m_axi port=cell_ID_DRAM offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=LUT_DRAM offset=slave bundle=gmem4

// in runtime (should from DRAM)
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_0 offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_1 offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_2 offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_3 offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_0  offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_1  offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_2  offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_3  offset=slave bundle=gmem12

// out
#pragma HLS INTERFACE m_axi port=out_DRAM  offset=slave bundle=gmem13

#pragma HLS dataflow

////////////////////     First Half: ADC     ////////////////////

    hls::stream<int> s_nlist_PQ_codes_start_addr;
#pragma HLS stream variable=s_nlist_PQ_codes_start_addr depth=256

    hls::stream<int> s_nlist_vec_ID_start_addr; // the top 10 numbers
#pragma HLS stream variable=s_nlist_vec_ID_start_addr depth=256
    
    hls::stream<int> s_nlist_num_vecs;
#pragma HLS stream variable=s_nlist_num_vecs depth=256

    load_nlist_init(
        nlist,
        nlist_init,
        s_nlist_PQ_codes_start_addr,
        s_nlist_vec_ID_start_addr,
        s_nlist_num_vecs);

    hls::stream<int> s_cell_ID_get_cell_addr_and_size;
#pragma HLS stream variable=s_cell_ID_get_cell_addr_and_size depth=256
    
    hls::stream<int> s_cell_ID_load_PQ_codes;
#pragma HLS stream variable=s_cell_ID_load_PQ_codes depth=256

    load_cell_ID(
        query_num,
        nprobe,
        cell_ID_DRAM,
        s_cell_ID_get_cell_addr_and_size,
        s_cell_ID_load_PQ_codes);

    hls::stream<int> s_scanned_entries_every_cell;
#pragma HLS stream variable=s_scanned_entries_every_cell depth=256
// #pragma HLS resource variable=s_scanned_entries_every_cell core=FIFO_SRL
    
    hls::stream<int> s_last_valid_PE_ID;
#pragma HLS stream variable=s_last_valid_PE_ID depth=256
// #pragma HLS resource variable=s_last_valid_PE_ID core=FIFO_SRL
    
    hls::stream<int> s_start_addr_every_cell;
#pragma HLS stream variable=s_start_addr_every_cell depth=256
// #pragma HLS resource variable=s_start_addr_every_cell core=FIFO_SRL
    
    hls::stream<int> s_control_iter_num_per_query;
#pragma HLS stream variable=s_control_iter_num_per_query depth=256
// #pragma HLS resource variable=s_control_iter_num_per_query core=FIFO_SRL
    
    get_cell_addr_and_size(
        // in init
        query_num, 
	    nlist,
        nprobe,
        s_nlist_PQ_codes_start_addr,
        s_nlist_num_vecs,
        // in runtime
        s_cell_ID_get_cell_addr_and_size,
        // out
        s_scanned_entries_every_cell,
        s_last_valid_PE_ID,
        s_start_addr_every_cell,
        s_control_iter_num_per_query);

    hls::stream<int> s_scanned_entries_every_cell_ADC[ADC_PE_NUM];
#pragma HLS stream variable=s_scanned_entries_every_cell_ADC depth=256
#pragma HLS array_partition variable=s_scanned_entries_every_cell_ADC complete
// #pragma HLS resource variable=s_scanned_entries_every_cell_ADC core=FIFO_SRL

    hls::stream<int> s_scanned_entries_every_cell_load_PQ_codes;
#pragma HLS stream variable=s_scanned_entries_every_cell_load_PQ_codes depth=256
// #pragma HLS resource variable=s_scanned_entries_every_cell_load_PQ_codes core=FIFO_SRL

    replicate_s_scanned_entries_every_cell(
        // in
        query_num,
        nprobe,
        s_scanned_entries_every_cell,
        // out
        s_scanned_entries_every_cell_ADC,
        s_scanned_entries_every_cell_load_PQ_codes);

    hls::stream<PQ_in_t> s_PQ_codes[ADC_PE_NUM];
#pragma HLS stream variable=s_PQ_codes depth=8
#pragma HLS array_partition variable=s_PQ_codes complete
// #pragma HLS resource variable=s_PQ_codes core=FIFO_SRL

    load_PQ_codes(
        // in init
        query_num, 
        nprobe,
        // in runtime
        s_cell_ID_load_PQ_codes,
        s_scanned_entries_every_cell_load_PQ_codes,
        s_last_valid_PE_ID,
        s_start_addr_every_cell,
        PQ_codes_DRAM_0,
        PQ_codes_DRAM_1,
        PQ_codes_DRAM_2,
        PQ_codes_DRAM_3,
        // out
        s_PQ_codes);

    hls::stream<PQ_out_t> s_PQ_result[ADC_PE_NUM];
#pragma HLS stream variable=s_PQ_result depth=8
#pragma HLS array_partition variable=s_PQ_result complete
// #pragma HLS resource variable=s_PQ_result core=FIFO_SRL

    hls::stream<distance_LUT_parallel_t> s_distance_LUT[ADC_PE_NUM + 1];
#pragma HLS stream variable=s_distance_LUT depth=8
#pragma HLS array_partition variable=s_distance_LUT complete
// #pragma HLS resource variable=s_distance_LUT core=FIFO_SRL

    // systolic array of distance LUT communication
    load_distance_LUT(
        query_num, 
        nprobe,
        LUT_DRAM, // query_num * nprobe * 256 * M
        s_distance_LUT[0]);

    for (int s = 0; s < ADC_PE_NUM; s++) {
#pragma HLS unroll
        PQ_lookup_computation(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT[s],
            s_PQ_codes[s],
            s_scanned_entries_every_cell_ADC[s],
            // output streams
            s_distance_LUT[s + 1],
            s_PQ_result[s]);
    }

    dummy_distance_LUT_consumer(
        query_num, 
        nprobe,
        s_distance_LUT[ADC_PE_NUM]);

////////////////////     Second Half: K-Selection     ////////////////////

    hls::stream<result_t> s_output; // the topK numbers
#pragma HLS stream variable=s_output depth=256

    hierarchical_priority_queue( 
        query_num, 
        nlist,
        s_nlist_vec_ID_start_addr,
        s_control_iter_num_per_query, 
        s_PQ_result,
        vec_ID_DRAM_0,
        vec_ID_DRAM_1,
        vec_ID_DRAM_2,
        vec_ID_DRAM_3,
        s_output);

    write_result(
        query_num, 
        s_output, 
        out_DRAM);
}

}
