#include <hls_stream.h>

#include "constants.hpp"
#include "hierarchical_priority_queue.hpp"
#include "types.hpp"

// #include <stdio.h>

void dummy_iter_num_per_query_sender(
    int query_num,
    const int iter_num_per_query_per_ADC_PE,
    hls::stream<int> (&s_control_iter_num_per_query)[2]) {

    for (int query_id = 0; query_id < query_num; query_id++) {
        s_control_iter_num_per_query[0].write(iter_num_per_query_per_ADC_PE);
        s_control_iter_num_per_query[1].write(iter_num_per_query_per_ADC_PE);
    }
}

void dummy_PQ_result_sender(
    int query_num,
    hls::stream<int> &s_control_iter_num_per_query,
    hls::stream<PQ_out_t> (&s_sorted_PQ_result)[ADC_PE_NUM]) {

    PQ_out_t sorted_array[ADC_PE_NUM];
#pragma HLS array_partition variable=sorted_array complete

    for (int s = 0; s < ADC_PE_NUM; s++) {
#pragma HLS UNROLL
        sorted_array[s].cell_ID = s;
        sorted_array[s].offset = s;
        sorted_array[s].dist = s;
    }

    for (int query_id = 0; query_id < query_num; query_id++) {

        int iter_num_per_query = s_control_iter_num_per_query.read();

        for (int iter = 0; iter < iter_num_per_query; iter++) {
#pragma HLS pipeline II=1

            for (int s = 0; s < ADC_PE_NUM; s++) {
#pragma HLS UNROLL
                s_sorted_PQ_result[s].write(sorted_array[s]);
            }
        }
    }
}

void load_nlist_vec_ID_start_addr(
    int nlist,
    int* nlist_vec_ID_start_addr,
    hls::stream<int> &s_nlist_vec_ID_start_addr) {

    for (int i = 0; i < nlist; i++) {
#pragma HLS pipeline
        s_nlist_vec_ID_start_addr.write(nlist_vec_ID_start_addr[i]);
    }
}

void write_result(
    int query_num,
    hls::stream<result_t> &output_stream, 
    ap_uint<64>* output) {

    // only write the last iteration
    for (int i = 0; i < (query_num - 1) * PRIORITY_QUEUE_LEN_L2; i++) {
        output_stream.read();
    }

    for (int i = 0; i < PRIORITY_QUEUE_LEN_L2; i++) {
#pragma HLS pipeline II=1
        result_t raw_output = output_stream.read();
        ap_uint<64> reg;
        int vec_ID = raw_output.vec_ID;
        float dist = raw_output.dist;
        reg.range(31, 0) = *((ap_uint<32>*) (&vec_ID));
        reg.range(63, 32) = *((ap_uint<32>*) (&dist));
        output[i] = reg;
    }
}


extern "C" {

void vadd(  
    int* nlist_vec_ID_start_addr,
    ap_uint<64>* vec_ID_DRAM_0,
    ap_uint<64>* vec_ID_DRAM_1,
    ap_uint<64>* vec_ID_DRAM_2,
    ap_uint<64>* vec_ID_DRAM_3,
    ap_uint<64>* out_DRAM,
    int query_num,
    int nlist,
    int iter_num_per_query_per_ADC_PE
    )
{
#pragma HLS INTERFACE m_axi port=nlist_vec_ID_start_addr  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_0  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_1  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_2  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_3  offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=out_DRAM  offset=slave bundle=gmem5

#pragma HLS dataflow

    hls::stream<int> s_control_iter_num_per_query[2];
#pragma HLS array_partition variable=s_control_iter_num_per_query complete
#pragma HLS stream variable=s_control_iter_num_per_query depth=2
#pragma HLS RESOURCE variable=s_control_iter_num_per_query core=FIFO_SRL

    dummy_iter_num_per_query_sender(
        query_num,
        iter_num_per_query_per_ADC_PE,
        s_control_iter_num_per_query);


    // 16 streams in sorted order, only need the top 10 
    hls::stream<PQ_out_t> s_sorted_PQ_result[ADC_PE_NUM];
#pragma HLS stream variable=s_sorted_PQ_result depth=8
#pragma HLS array_partition variable=s_sorted_PQ_result complete
#pragma HLS RESOURCE variable=s_sorted_PQ_result core=FIFO_SRL

    dummy_PQ_result_sender(
        query_num,
        s_control_iter_num_per_query[0],
        s_sorted_PQ_result);

    hls::stream<int> s_nlist_vec_ID_start_addr; // the top 10 numbers
#pragma HLS stream variable=s_nlist_vec_ID_start_addr depth=2

    load_nlist_vec_ID_start_addr(
        nlist,
        nlist_vec_ID_start_addr,
        s_nlist_vec_ID_start_addr);

    hls::stream<result_t> s_output; // the top 10 numbers
#pragma HLS stream variable=s_output depth=256

    hierarchical_priority_queue( 
        query_num, 
        nlist,
        s_nlist_vec_ID_start_addr,
        s_control_iter_num_per_query[1], 
        s_sorted_PQ_result,
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