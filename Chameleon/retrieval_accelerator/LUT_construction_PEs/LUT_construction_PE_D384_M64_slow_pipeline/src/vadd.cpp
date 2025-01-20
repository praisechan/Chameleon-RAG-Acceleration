#include "constants.hpp"
#include "types.hpp"

#include "LUT_construction.hpp"

void load_query_vectors(
    int query_num,
    ap_uint<512>* DRAM_query_vector,
    hls::stream<ap_uint<512>>& s_query_vectors) {

    // query format: store in 512-bit packets, pad 0 for the last packet if needed
    const int size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    
    for (int query_id = 0; query_id < query_num; query_id++) {

        // load query
        for (int i = 0; i < size_query_vector; i++) {

            s_query_vectors.write(DRAM_query_vector[query_id * size_query_vector + i]);
        }
    }
}

void load_center_vectors(
    int query_num,
    int nprobe,
    ap_uint<512>* DRAM_center_vector,
    hls::stream<ap_uint<512>>& s_center_vectors) {

    // center vector format: store in 512-bit packets, pad 0 for the last packet if needed
    const int size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            // load center vector
            for (int i = 0; i < size_center_vector; i++) {

                s_center_vectors.write(
                    DRAM_center_vector[(query_id * nprobe + nprobe_id) * size_center_vector + i]);
            }
        }
    }
}

void load_PQ_quantizer(
    float* DRAM_PQ_quantizer,
    hls::stream<float> &s_product_quantizer_init) {

    // load PQ quantizer centers from HBM
    for (int i = 0; i < LUT_ENTRY_NUM * D; i++) {
#pragma HLS pipeline II=1
        s_product_quantizer_init.write(DRAM_PQ_quantizer[i]);
    }
}

template<const int m>
void consume_and_write(
    int query_num,
    int nprobe,
    hls::stream<distance_LUT_parallel_t>& s_result, 
    ap_uint<512>* results_out);
    

template<>
void consume_and_write<16>(
    int query_num,
    int nprobe,
    hls::stream<distance_LUT_parallel_t>& s_result, 
    ap_uint<512>* DRAM_out) {
    
    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            for (int k = 0; k < LUT_ENTRY_NUM; k++) {
#pragma HLS pipeline II=1

                distance_LUT_parallel_t reg_in = s_result.read();
                ap_uint<512> reg_out; 
                for (int j = 0; j < 16; j++) {
#pragma HLS UNROLL
                    float dist_float = reg_in.dist[j];
                    ap_uint<32> dist_uint = *((ap_uint<32>*) (&dist_float));
                    reg_out.range(32 * j + 31, 32 * j) = dist_uint;
                }
                DRAM_out[(query_id * nprobe + nprobe_id) * LUT_ENTRY_NUM + k] = reg_out;
            }
        }
    }
}

template<>
void consume_and_write<32>(
    int query_num,
    int nprobe,
    hls::stream<distance_LUT_parallel_t>& s_result, 
    ap_uint<512>* DRAM_out) {
    
    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            for (int k = 0; k < LUT_ENTRY_NUM; k++) {
#pragma HLS pipeline II=2

                // every row of LUT = 2 * ap_uint<512> 
                distance_LUT_parallel_t reg_in = s_result.read();

                ap_uint<512> reg_out_A; 
                for (int j = 0; j < 16; j++) {
#pragma HLS UNROLL
                    float dist_float = reg_in.dist[j];
                    ap_uint<32> dist_uint = *((ap_uint<32>*) (&dist_float));
                    reg_out_A.range(32 * j + 31, 32 * j) = dist_uint;
                }
                DRAM_out[2 * ((query_id * nprobe + nprobe_id) * LUT_ENTRY_NUM + k)] = reg_out_A;

                ap_uint<512> reg_out_B; 
                for (int j = 0; j < 16; j++) {
#pragma HLS UNROLL
                    float dist_float = reg_in.dist[j + 16];
                    ap_uint<32> dist_uint = *((ap_uint<32>*) (&dist_float));
                    reg_out_B.range(32 * j + 31, 32 * j) = dist_uint;
                }
                DRAM_out[2 * ((query_id * nprobe + nprobe_id) * LUT_ENTRY_NUM + k) + 1] = reg_out_B;
            }
        }
    }
}


template<>
void consume_and_write<64>(
    int query_num,
    int nprobe,
    hls::stream<distance_LUT_parallel_t>& s_result, 
    ap_uint<512>* DRAM_out) {
    
    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            for (int k = 0; k < LUT_ENTRY_NUM; k++) {
#pragma HLS pipeline II=4

                // every row of LUT = 2 * ap_uint<512> 
                distance_LUT_parallel_t reg_in = s_result.read();

                ap_uint<512> reg_out_A; 
                for (int j = 0; j < 16; j++) {
#pragma HLS UNROLL
                    float dist_float = reg_in.dist[j];
                    ap_uint<32> dist_uint = *((ap_uint<32>*) (&dist_float));
                    reg_out_A.range(32 * j + 31, 32 * j) = dist_uint;
                }
                DRAM_out[4 * ((query_id * nprobe + nprobe_id) * LUT_ENTRY_NUM + k)] = reg_out_A;

                ap_uint<512> reg_out_B; 
                for (int j = 0; j < 16; j++) {
#pragma HLS UNROLL
                    float dist_float = reg_in.dist[j + 16];
                    ap_uint<32> dist_uint = *((ap_uint<32>*) (&dist_float));
                    reg_out_B.range(32 * j + 31, 32 * j) = dist_uint;
                }
                DRAM_out[4 * ((query_id * nprobe + nprobe_id) * LUT_ENTRY_NUM + k) + 1] = reg_out_B;

                ap_uint<512> reg_out_C; 
                for (int j = 0; j < 16; j++) {
#pragma HLS UNROLL
                    float dist_float = reg_in.dist[j + 32];
                    ap_uint<32> dist_uint = *((ap_uint<32>*) (&dist_float));
                    reg_out_C.range(32 * j + 31, 32 * j) = dist_uint;
                }
                DRAM_out[4 * ((query_id * nprobe + nprobe_id) * LUT_ENTRY_NUM + k) + 2] = reg_out_C;

                ap_uint<512> reg_out_D; 
                for (int j = 0; j < 16; j++) {
#pragma HLS UNROLL
                    float dist_float = reg_in.dist[j + 48];
                    ap_uint<32> dist_uint = *((ap_uint<32>*) (&dist_float));
                    reg_out_C.range(32 * j + 31, 32 * j) = dist_uint;
                }
                DRAM_out[4 * ((query_id * nprobe + nprobe_id) * LUT_ENTRY_NUM + k) + 3] = reg_out_D;
            }
        }
    }
}



extern "C" {

void vadd(  
    // init
    int query_num,
    int nprobe,
    float* DRAM_PQ_quantizer,
    // runtime input
    ap_uint<512>* DRAM_query_vector,
    ap_uint<512>* DRAM_center_vector,
    // output
    ap_uint<512>* DRAM_out
    )
{
#pragma HLS INTERFACE m_axi port=DRAM_PQ_quantizer  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=DRAM_query_vector  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=DRAM_center_vector  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=DRAM_out  offset=slave bundle=gmem3


#pragma HLS dataflow

    hls::stream<float> s_product_quantizer_init;
#pragma HLS stream variable=s_product_quantizer_init depth=512
   
    load_PQ_quantizer(
        DRAM_PQ_quantizer,
        s_product_quantizer_init); 

    hls::stream<ap_uint<512>> s_query_vectors;
#pragma HLS stream variable=s_query_vectors depth=512

    hls::stream<ap_uint<512>> s_center_vectors;
#pragma HLS stream variable=s_center_vectors depth=512

    load_query_vectors(
        query_num,
        DRAM_query_vector, 
        s_query_vectors);

    load_center_vectors(
        query_num, 
        nprobe,
        DRAM_center_vector, 
        s_center_vectors);

    hls::stream<distance_LUT_parallel_t> s_distance_LUT;
#pragma HLS stream variable=s_distance_LUT depth=512
// #pragma HLS resource variable=s_distance_LUT core=FIFO_SRL

    LUT_construction_wrapper(
        // init
        query_num,
        nprobe, 
        s_product_quantizer_init,
        // runtime input from network
        s_query_vectors,
        s_center_vectors,
        // output
        s_distance_LUT);

    consume_and_write<M>(
        query_num,
        nprobe,
        s_distance_LUT, 
        DRAM_out);
}
}


