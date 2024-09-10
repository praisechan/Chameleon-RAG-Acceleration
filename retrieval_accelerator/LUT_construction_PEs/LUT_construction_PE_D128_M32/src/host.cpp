#include "host.hpp"

#include "constants.hpp"
// Wenqi: seems 2022.1 somehow does not support linking ap_uint.h to host?
// #include "ap_uint.h"


int main(int argc, char** argv)
{
    cl_int err;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr 
    // is used if it is properly aligned. when not aligned, runtime had no choice but to create
    // its own host side buffer. So it is recommended to use this allocator if user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
    // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR 

    std::cout << "Allocating memory" << std::endl;

    size_t query_num = 1000;
    size_t nprobe = 32;

    size_t product_quantizer_bytes = D * LUT_ENTRY_NUM * sizeof(float);
    std::vector<float ,aligned_allocator<float >> product_quantizer(product_quantizer_bytes / sizeof(float));
    // reshape for software
    std::vector<float ,aligned_allocator<float >> product_quantizer_reshaped(product_quantizer_bytes / sizeof(float));

    // query format: store in 512-bit packets, pad 0 for the last packet if needed
    const size_t size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    size_t query_vector_bytes = query_num * size_query_vector * 64;
    std::vector<float ,aligned_allocator<float >> query_vectors(query_vector_bytes / sizeof(float));

    const size_t size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    size_t center_vector_bytes = query_num * nprobe * size_center_vector * 64;
    std::vector<float ,aligned_allocator<float >> center_vectors(center_vector_bytes / sizeof(float));

    size_t out_bytes = query_num * nprobe * LUT_ENTRY_NUM * M * sizeof(float);
    std::vector<float ,aligned_allocator<float>> out(out_bytes / sizeof(float));


    // init 
    // M * 256 * D/M
    // std::cout << "1\n";
    for (int i = 0; i < D * LUT_ENTRY_NUM; i++) {
        product_quantizer[i] = i % LUT_ENTRY_NUM;
    }    
    // reshape product quantizer: M * 256 * (D/M) -> 256 * D == 256 * M * (D / M)
    for (int j = 0; j < 256; j++) {
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < D / M; k++) {
                product_quantizer_reshaped[j * D + i * D / M + k] =
                    product_quantizer[i * 256 * D / M + j * D / M + k];
            }
        }
    }

    // std::cout << "2\n";
    float query_vec_data[D] =  {9, 26, -34, -2, -83, -17, 6, 23, -43, -4, 13, 26, -53, -27, -68, 74, 11, 
        53, -17, -22, 64, -4, -32, -51, -45, 95, -98, -16, -61, -34, -16, -53, 89, 76, 35, 5, -1, 
        24, -8, 80, 2, -3, 18, -6, 55, -66, -24, 68, 31, 6, -31, -36, -25, 62, -42, 38, -78, 46, -85, 
        55, 58, 80, -30, 9, 15, 54, 34, -86, 3, 82, 99, 29, -57, 86, 1, 83, -75, 78, -44, -51, -88, 
        -82, -99, -49, 72, 17, -52, -44, 77, -14, -97, -33, 39, 49, 3, -2, -97, 39, -97, -94, -91, 
        -13, -86, 21, 98, 40, 82, -73, -62, 45, 89, 10, -26, -1, 16, -1, 93, -53, -84, 33, 4, 9, 
        -14, -54, -85, -41, -60, 53};

    for (int i = 0; i < query_num; i++) {
        for (int j = 0; j < D; j++) {
            query_vectors[i * D + j] = query_vec_data[j];
        }
    }

    // std::cout << "3\n";
    float center_vec_data[D] = {55, 1, 71, -11, -58, 56, 10, -21, 22, -40, -55, 65, 67, -89, -23, 97, 
        -16, -66, -34, 7, 4, -96, -92, -38, 2, -84, -59, 38, -28, -56, -65, 18, 12, 76, 79, -47, 
        22, 25, 50, 63, 84, 50, -15, 34, -56, -56, -51, 71, 39, -79, -10, 10, -24, -38, -80, 17, 
        94, -50, -32, -21, -31, 32, -49, 59, 86, -3, 38, -9, -75, -96, 39, 56, -95, 21, 11, -29, 
        26, 90, 85, 90, 40, -59, -57, -53, 53, -85, -16, 17, 45, -74, -68, 42, 10, -24, -28, -65, 
        36, -40, 55, 26, -85, 58, -20, 81, 8, -90, 20, 0, 25, 99, 65, 72, -11, 7, -86, 86, 48, 6, 
        -23, -25, -74, 91, 54, 21, 52, -38, 9, 96};

    for (int i = 0; i < query_num * nprobe; i++) {
        for (int j = 0; j < D; j++) {
            center_vectors[i * D + j] = center_vec_data[j];
        }
    }


    // Compute software result (for the first query vec & first nprobe)
    std::vector<float> diff_vec(D); // query_vec - centroid 
    std::vector<float> LUT(D * LUT_ENTRY_NUM); //(256, M)

    for (size_t d = 0; d < D; d++) {
        diff_vec[d] = query_vectors[d] - center_vectors[d];
    }
    
    for (size_t row_id = 0; row_id < LUT_ENTRY_NUM; row_id++) {

        for (size_t m = 0; m < M; m++) {

            float dist = 0;
            for (size_t c = 0; c < D / M; c++) {
                dist += 
                    (diff_vec[m * D / M + c] - product_quantizer_reshaped[row_id * D + m * D / M + c]) * 
                    (diff_vec[m * D / M + c] - product_quantizer_reshaped[row_id * D + m * D / M + c]);
            }
            LUT[row_id * M + m] = dist;
            // std::cout << "row: " << row_id << " col (m): " << m << " dist: " << dist << std::endl;
        }
    }
    

// OPENCL HOST CODE AREA START

    std::vector<cl::Device> devices = get_devices();
    cl::Device device = devices[0];
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    //Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue q(context, device);

    // Import XCLBIN
    xclbin_file_name = argv[1];
    cl::Program::Binaries vadd_bins = import_binary_file();

    // Program and Kernel
    devices.resize(1);
    cl::Program program(context, devices, vadd_bins);
    cl::Kernel krnl_vector_add(program, "vadd");

	// ------------------------------------------------------------------
	// Create Buffers in Global Memory to store data
	//             o) buffer_in1 - stores source_in1
	//             o) buffer_in2 - stores source_in2
	//             o) buffer_ouput - stores Results
	// ------------------------------------------------------------------	
	
	// Allocate Global Memory for source_in1
    OCL_CHECK(err, cl::Buffer buffer_product_quantizer   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            product_quantizer_bytes, product_quantizer.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_query_vector   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            query_vector_bytes, query_vectors.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_center_vector   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            center_vector_bytes, center_vectors.data(), &err));

	// Allocate Global Memory for sourcce_hw_results
    OCL_CHECK(err, cl::Buffer buffer_out(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            out_bytes, out.data(), &err));

    OCL_CHECK(err, err = krnl_vector_add.setArg(0, int(query_num)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(1, int(nprobe)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(2, buffer_product_quantizer));
    OCL_CHECK(err, err = krnl_vector_add.setArg(3, buffer_query_vector));
    OCL_CHECK(err, err = krnl_vector_add.setArg(4, buffer_center_vector));
    OCL_CHECK(err, err = krnl_vector_add.setArg(5, buffer_out));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        buffer_product_quantizer, buffer_query_vector, buffer_center_vector},0/* 0 means from host*/));

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out},CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

// OPENCL HOST CODE AREA END
    std::cout << "Comparing Results..." << std::endl;
    bool match = true;
    int count = 0;
    int correct_count = 0;

    // check the first distance LUT
    for (int i = 0 ; i < LUT_ENTRY_NUM; i++) {
        for (int j = 0; j < M; j++) {

            count++;
            float hw_result = out[i * M + j];
            float sw_result = LUT[i * M + j];

            if ((hw_result - sw_result <= 0.01) && (hw_result - sw_result >= -0.01) &&
                (hw_result / sw_result <= 1.01) && (hw_result / sw_result >= 0.99)) {
                correct_count++;
            } else {
                printf("MISMATCH: dist_table(%d, %d): hw %f sw %f\n", i, j, hw_result, sw_result);
            }
        }
    }
    std::cout << "Result match rate " << correct_count / count << std::endl; 

    std::cout << "TEST FINISHED" << std::endl; 

    return  0;
}
