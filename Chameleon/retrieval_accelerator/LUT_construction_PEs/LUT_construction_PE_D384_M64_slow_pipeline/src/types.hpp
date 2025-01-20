#pragma once

#include <ap_int.h>
#include <hls_stream.h>

#include "constants.hpp"

// the distance LUT input type
typedef struct {
    // each distance LUT has K=256 such row
    // each distance_LUT_PQ16_t is the content of a single row (16 floats)
    float dist[M];
} distance_LUT_parallel_t;
