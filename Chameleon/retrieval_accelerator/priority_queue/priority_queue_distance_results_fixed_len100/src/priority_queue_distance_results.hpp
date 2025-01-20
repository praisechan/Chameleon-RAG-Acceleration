#pragma once 

// #include <type_traits>
#include "constants.hpp"
// NOTES: 1. use 2020.2 
// 2. Use constructor to partition member array / include pragma 
//    ERROR: [v++ 214-122] '#pragma HLS' is only allowed in function scope: /mnt/scratch/wenqi/FPGA-ANNS/building_blocks/class_priority_queue/src/priority_queue.hpp:23
// enum class Order { Collect_smallest, Collect_largest };

template<typename T, const int queue_size, Order order> 
class Priority_queue;

template<const int queue_size> 
class Priority_queue<single_PQ_result, queue_size, Collect_smallest> {

    public: 

        Priority_queue() {
#pragma HLS inline
        }

        template<const int query_num, const int iter_num_per_query>
        void insert_wrapper(
            // hls::stream<int>& control_stream_iter_num_per_query,
            hls::stream<single_PQ_result> &s_input, 
            hls::stream<single_PQ_result> &s_output) {
            
            // smaller value swapped to the right (larger index ID)
            // queue[0] used to store input value
            single_PQ_result queue[queue_size + 1];
#pragma HLS array_partition variable=queue complete

            for (int query_id = 0; query_id < query_num; query_id++) {
                // Fixed, originally "const int iter_num"
                // int iter_num = control_stream_iter_num_per_query.read();

                // init
                for (int i = 0; i < queue_size + 1; i++) {
#pragma HLS UNROLL
                    queue[i].vec_ID = -1;
                    queue[i].dist = LARGE_NUM;
                }

                // insert: 
                for (int i = 0; i < iter_num_per_query; i++) {
#pragma HLS pipeline II=1
                    // single_PQ_result reg = s_input.read();
                    // queue[0] = queue[0].dist < reg.dist? queue[0] : reg;

                    queue[0] = s_input.read();

                    // compare_swap_array_step_A(queue);
                    // compare_swap_array_step_B(queue);

//                     for (int j = 0; j < (queue_size + 1) / 2; j++) {
// #pragma HLS UNROLL
//                         if (queue[2 * j].dist < queue[2 * j + 1].dist) {
//                             single_PQ_result regA = queue[2 * j];
//                             single_PQ_result regB = queue[2 * j + 1];
//                             queue[2 * j] = regB;
//                             queue[2 * j + 1] = regA;
//                         }
//                     }
//                     for (int j = 0; j < queue_size / 2; j++) {
// #pragma HLS UNROLL
//                         if (queue[2 * j + 1].dist < queue[2 * j + 2].dist) {
//                             single_PQ_result regA = queue[2 * j + 1];
//                             single_PQ_result regB = queue[2 * j + 2];
//                             queue[2 * j + 1] = regB;
//                             queue[2 * j + 2] = regA;
//                         }
//                     }

                    for (int j = 0; j < (queue_size) / 2; j++) {
#pragma HLS UNROLL
                        bool swap_A = queue[2 * j].dist < queue[2 * j + 1].dist? 1: 0;
#pragma HLS bind_op variable=swap_A impl=fulldsp
                        if (swap_A) {
                        // if (queue[2 * j].dist < queue[2 * j + 1].dist) {
                            single_PQ_result regA = queue[2 * j];
                            single_PQ_result regB = queue[2 * j + 1];
                            queue[2 * j] = regB;
                            queue[2 * j + 1] = regA;
                        }
                        bool swap_B = queue[2 * j + 1].dist < queue[2 * j + 2].dist? 1 : 0;
#pragma HLS bind_op variable=swap_B impl=fulldsp
                        // if (queue[2 * j + 1].dist < queue[2 * j + 2].dist) {
                        if (swap_B) {
                            single_PQ_result regA = queue[2 * j + 1];
                            single_PQ_result regB = queue[2 * j + 2];
                            queue[2 * j + 1] = regB;
                            queue[2 * j + 2] = regA;
                        }
                    }
                }

                // write
                for (int i = 0; i < queue_size; i++) {
#pragma HLS pipeline II=1
                    s_output.write(queue[i]);
//                     single_PQ_result out_reg = queue[queue_size];
//                     for (int j = queue_size; j > 0; j--) {
// #pragma HLS UNROLL
//                         queue[j] = queue[j - 1];
//                     }
//                     s_output.write(out_reg);
                }
            }
        }


    private:
    
        void compare_swap(single_PQ_result* array, int idxA, int idxB) {
            // if smaller -> swap to right
            // note: idxA must < idxB
#pragma HLS inline
            if (array[idxA].dist < array[idxB].dist) {
                single_PQ_result regA = array[idxA];
                single_PQ_result regB = array[idxB];
                array[idxA] = regB;
                array[idxB] = regA;
            }
        }

        void compare_swap_array_step_A(single_PQ_result* array) {
            // start from idx 0, odd-even swap
            // the entire physical queue size scope: queue_size + 1
#pragma HLS inline
            for (int j = 0; j < (queue_size + 1) / 2; j++) {
#pragma HLS UNROLL
                compare_swap(array, 2 * j, 2 * j + 1);
            }
        }
                    
        void compare_swap_array_step_B(single_PQ_result* array) {
            // start from idx 1, odd-even swap
            // does not involve reg[0], the entire logical queue size scope: queue_size
#pragma HLS inline
            for (int j = 0; j < queue_size / 2; j++) {
#pragma HLS UNROLL
                compare_swap(array, 2 * j + 1, 2 * j + 2);
            }
        }
};
