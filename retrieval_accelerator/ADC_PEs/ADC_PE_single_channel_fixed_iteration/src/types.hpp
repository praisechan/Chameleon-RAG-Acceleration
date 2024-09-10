#pragma once

#include <ap_int.h>

#include "constants.hpp"

// The input to an ADC computation PE
typedef struct {
    bool valid;  // if invalid, the vector is dummy data, the compute PE
		 //  should handle it respectively
    int cell_ID; // Voronoi cell ID: [0, nlist - 1]
    int offset;  // the i-th vector in the cell in the memory channel
                 // different memory channels can have the same offsets
		 // the offset will be used to lookup the real vecID in the channel
    ap_uint<8> PQ_code[M];
} PQ_in_t;

// The output of an ADC computation PE
 typedef struct {
    int cell_ID; // Voronoi cell ID: [0, nlist - 1]
    int offset;  // the i-th vector in the cell in the memory channel
                 // different memory channels can have the same offsets
		 // the offset will be used to lookup the real vecID in the channel
    float dist;
} PQ_out_t; 

typedef struct {
    // each distance LUT has K=256 such row
    // each distance_LUT_PQ16_t is the content of a single row (16 floats)
    float dist[M];
} distance_LUT_parallel_t;

// typedef ap_uint<512> t_axi;