/* Consumes PQ Codes located in a single channel, and compute PQ distance */

#include <hls_stream.h>

#include "constants.hpp"
#include "types.hpp"


void load_PQ_codes(
    int query_num, 
    int nprobe,
    int compute_iter_per_PE, 
    const ap_uint<512>* DRAM,
    hls::stream<PQ_in_t> (&s_single_PQ)[ADC_PE_PER_CHANNEL]
) {

    // TODO: for real test, should have a table that maps cell_ID to start_address
    //
    //

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            for (int entry_id = 0; entry_id < compute_iter_per_PE; entry_id++) {
#pragma HLS pipeline II=1
                ap_uint<512> PQ_reg_multi_channel = DRAM[compute_iter_per_PE];

                for (int s = 0; s < ADC_PE_PER_CHANNEL; s++) {
#pragma HLS unroll
                    PQ_in_t PQ_reg;
                    
                    // TODO: for real test, should check whether the last codes are valid
                    //
                    //
                    PQ_reg.valid = 1; 
                    // TODO: for real test, should use real cell ID
                    //
                    //
                    PQ_reg.cell_ID = nprobe_id; 
                    PQ_reg.offset = entry_id * ADC_PE_PER_CHANNEL + s;
                    // refer: https://github.com/WenqiJiang/FPGA-ANNS/blob/main/integrated_accelerator/entire-node-systolic-computation-without-FIFO-type-assignment-fine-grained-PE-with-queue-group-inlined/src/HBM_interconnections.hpp
                    for (int m = 0; m < M; m++) {
#pragma HLS unroll
                        PQ_reg.PQ_code[m] = PQ_reg_multi_channel.range(
                            s * M * 8 + m * 8 + 7, s * M * 8 + m * 8);
                    }
                    s_single_PQ[s].write(PQ_reg);
                }
            }
        }
    }
}

void dummy_scanned_entries_every_cell(
    int query_num, 
    int nprobe,
    int scanned_entries_every_cell_const,
    hls::stream<int> &s_scanned_entries_every_cell) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            s_scanned_entries_every_cell.write(scanned_entries_every_cell_const); 
        }
    }
}

void dummy_distance_LUT_sender(
    int query_num, 
    int nprobe,
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT) {

    distance_LUT_parallel_t dist_row;

    for (int i = 0; i < M; i++) {
#pragma HLS unroll
        dist_row.dist[i] = i;
    }

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            for (int row_id = 0; row_id < LUT_ENTRY_NUM; row_id++) {
#pragma HLS pipeline II=1
                s_distance_LUT.write(dist_row);
            }
        }
    }
    
}

void dummy_distance_LUT_consumer(
    int query_num, 
    int nprobe,
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT) {

    distance_LUT_parallel_t dist_row;

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            for (int row_id = 0; row_id < LUT_ENTRY_NUM; row_id++) {
#pragma HLS pipeline II=1
                dist_row = s_distance_LUT.read();
            }

        }
    }
}

void PQ_lookup_computation(
    int query_num, 
    int nprobe,
    // input streams
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT_in,
    hls::stream<PQ_in_t>& s_single_PQ,
    hls::stream<int>& s_scanned_entries_every_cell_PQ_lookup_computation,
    // output streams
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT_out,
    hls::stream<PQ_out_t>& s_single_PQ_result) {

    // Manual control on the double buffer
    float distance_LUT_A[M][LUT_ENTRY_NUM];
#pragma HLS array_partition variable=distance_LUT_A dim=1
#pragma HLS resource variable=distance_LUT_A core=RAM_1P_BRAM
    float distance_LUT_B[M][LUT_ENTRY_NUM];
#pragma HLS array_partition variable=distance_LUT_B dim=1
#pragma HLS resource variable=distance_LUT_B core=RAM_1P_BRAM

    QUERY_LOOP:
    for (int query_id = 0; query_id < query_num; query_id++) {
        
        NPROBE_LOOP:
        for (int nprobe_id = 0; nprobe_id < nprobe + 1; nprobe_id++) {

            // iter 0: read data only
            // middle iter: read + compute
            // final iter: compute only
            int scanned_entries_every_cell;
            if (nprobe_id > 0) {
                scanned_entries_every_cell = 
                    s_scanned_entries_every_cell_PQ_lookup_computation.read();
            }
            else {
                // for the first iteration, there's no computation, just randomly
                //  init these variables
                scanned_entries_every_cell = 0;
            }

            int max_iter = scanned_entries_every_cell > LUT_ENTRY_NUM?
                scanned_entries_every_cell : LUT_ENTRY_NUM;

            int nprobe_mod = nprobe_id % 2;

            for (int common_iter = 0; common_iter < max_iter; common_iter++) {
#pragma HLS pipeline II=1

                // Note! Use the if else clause such that A and B certainly
                //   won't have bank conflict in a single CC

                // even: load to buffer A, compute by buffer B
                if (nprobe_mod == 0) { 

                    // load part
                    // last iter not read 
                    if (nprobe_id < nprobe) { 

                        // load or not
                        if (common_iter < LUT_ENTRY_NUM) { 

                            distance_LUT_parallel_t dist_row = s_distance_LUT_in.read();
                            s_distance_LUT_out.write(dist_row);

                            // even: load to buffer A
                            for (int col_id = 0; col_id < M; col_id++) {
#pragma HLS UNROLL
                                distance_LUT_A[col_id][common_iter] = dist_row.dist[col_id]; 
                            }
                        }
                    }

                    // compute part
                    // first iter not compute
                    if (nprobe_id > 0) { 

                        // compute or not
                        if (common_iter < scanned_entries_every_cell) { 

                            PQ_in_t PQ_local = s_single_PQ.read();

                            PQ_out_t out; 
                            out.cell_ID = PQ_local.cell_ID;
                            out.offset = PQ_local.offset;
                            
                            out.dist = 0;
                            for (int b = 0; b < M; b++) {
#pragma HLS unroll
                                out.dist += distance_LUT_B[b][PQ_local.PQ_code[b]];
                            }       
                                
                            // for padded element, replace its distance by large number
                            if (PQ_local.valid == 0) {
                                out.cell_ID = -1;
                                out.offset = -1;
                                out.dist = LARGE_NUM;
                            }
                            s_single_PQ_result.write(out);
                        }
                    }
                }
                // odd: load from buffer B, compute by buffer A
                else {

                    // load part
                    // last iter not read 
                    if (nprobe_id < nprobe) { 

                        // load or not
                        if (common_iter < LUT_ENTRY_NUM) { 

                            distance_LUT_parallel_t dist_row = s_distance_LUT_in.read();
                            s_distance_LUT_out.write(dist_row);

                            // odd: load to buffer B
                            for (int col_id = 0; col_id < M; col_id++) {
#pragma HLS UNROLL
                                distance_LUT_B[col_id][common_iter] = dist_row.dist[col_id]; 
                            }
                        }
                    }

                    // compute part
                    // first iter not compute
                    if (nprobe_id > 0) { 

                        // compute or not
                        if (common_iter < scanned_entries_every_cell) { 

                            PQ_in_t PQ_local = s_single_PQ.read();

                            PQ_out_t out; 
                            out.cell_ID = PQ_local.cell_ID;
                            out.offset = PQ_local.offset;
                            
                            out.dist = 0;
                            for (int b = 0; b < M; b++) {
#pragma HLS unroll
                                out.dist += distance_LUT_A[b][PQ_local.PQ_code[b]];
                            }       
                            // for padded element, replace its distance by large number
                            if (PQ_local.valid == 0) {
                                out.cell_ID = -1;
                                out.offset = -1;
                                out.dist = LARGE_NUM;
                            }
                            s_single_PQ_result.write(out);
                        }
                    }
                }
            }
        }
    }
}

void write_result(
    int query_num, 
    int nprobe,
    int compute_iter_per_PE,
    hls::stream<PQ_out_t> (&s_result)[ADC_PE_PER_CHANNEL]
    , ap_uint<96>* results_out) {
    
    PQ_out_t reg[ADC_PE_PER_CHANNEL];

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            for (int entry_id = 0; entry_id < compute_iter_per_PE; entry_id++) {
#pragma HLS pipeline II=1

                for (int s = 0; s < ADC_PE_PER_CHANNEL; s++) {
#pragma HLS unroll
                    reg[s] = s_result[s].read();
                }
            }
        }
    }

    // only write the last value to out
    int cell_ID = reg[0].cell_ID;
    int offset = reg[0].offset;
    float dist = reg[0].dist;
    ap_uint<32> cell_ID_ap = *((ap_uint<32>*) (&cell_ID));
    ap_uint<32> offset_ap = *((ap_uint<32>*) (&offset));
    ap_uint<32> dist_ap = *((ap_uint<32>*) (&dist));
    results_out[0].range(31, 0) = cell_ID_ap;
    results_out[0].range(63, 32) = offset_ap;
    results_out[0].range(96, 64) = dist_ap;
}


extern "C" {


void vadd(  
    const ap_uint<512>* in, 
    ap_uint<96>* out,
    int query_num, 
    int nprobe,
    int compute_iter_per_PE
    )
{
#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1

#pragma HLS dataflow

    hls::stream<PQ_in_t> s_PQ_codes[ADC_PE_PER_CHANNEL];
#pragma HLS stream variable=s_PQ_codes depth=8
#pragma HLS array_partition variable=s_PQ_codes complete
// #pragma HLS resource variable=s_PQ_codes core=FIFO_SRL

    hls::stream<PQ_out_t> s_single_PQ_result[ADC_PE_PER_CHANNEL];
#pragma HLS stream variable=s_single_PQ_result depth=8
#pragma HLS array_partition variable=s_single_PQ_result complete
// #pragma HLS resource variable=s_single_PQ_result core=FIFO_SRL

    hls::stream<int> s_scanned_entries_every_cell[ADC_PE_PER_CHANNEL];
#pragma HLS stream variable=s_scanned_entries_every_cell depth=8
#pragma HLS array_partition variable=s_scanned_entries_every_cell complete
// #pragma HLS resource variable=s_scanned_entries_every_cell core=FIFO_SRL

    hls::stream<distance_LUT_parallel_t> s_distance_LUT_in[ADC_PE_PER_CHANNEL];
#pragma HLS stream variable=s_distance_LUT_in depth=8
#pragma HLS array_partition variable=s_distance_LUT_in complete
// #pragma HLS resource variable=s_distance_LUT_in core=FIFO_SRL

    hls::stream<distance_LUT_parallel_t> s_distance_LUT_out[ADC_PE_PER_CHANNEL];
#pragma HLS stream variable=s_distance_LUT_out depth=8
#pragma HLS array_partition variable=s_PQ_codes complete
// #pragma HLS resource variable=s_distance_LUT_out core=FIFO_SRL

// #define COMPUTE_ITER_PER_PE (SCANNED_ENTRIES_PER_CELL / ADC_PE_PER_CHANNEL)
    load_PQ_codes(
        query_num, 
        nprobe,
        compute_iter_per_PE,
        in,
        s_PQ_codes);

    for (int s = 0; s < ADC_PE_PER_CHANNEL; s++) {
#pragma HLS unroll
    dummy_scanned_entries_every_cell(
        query_num, 
        nprobe,
        compute_iter_per_PE,
        s_scanned_entries_every_cell[s]);
    }

    for (int s = 0; s < ADC_PE_PER_CHANNEL; s++) {
#pragma HLS unroll
    dummy_distance_LUT_sender(
        query_num, 
        nprobe,
        s_distance_LUT_in[s]);
    }

////////////////////    Core Function Starts     //////////////////// 
    for (int s = 0; s < ADC_PE_PER_CHANNEL; s++) {
#pragma HLS unroll
        PQ_lookup_computation(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT_in[s],
            s_PQ_codes[s], 
            s_scanned_entries_every_cell[s],
            // output streams
            s_distance_LUT_out[s],
            s_single_PQ_result[s]);
    }

////////////////////    Core Function Ends     //////////////////// 

    for (int s = 0; s < ADC_PE_PER_CHANNEL; s++) {
#pragma HLS unroll
    dummy_distance_LUT_consumer(
        query_num, 
        nprobe,
        s_distance_LUT_out[s]);
    }

    write_result(
        query_num, 
        nprobe,
        compute_iter_per_PE,
        s_single_PQ_result, 
        out);
}

}