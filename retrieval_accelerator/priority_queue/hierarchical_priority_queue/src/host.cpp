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

    int nlist = NLIST_MAX;
    int query_num = QUERY_NUM;
    int iter_num_per_query_per_ADC_PE = 10000;

    int entries_per_cell = 100;

    size_t addr_table_size = nlist * sizeof(int);
    size_t in_bytes = nlist * entries_per_cell * sizeof(int);
    size_t out_bytes = 3 * TOPK;
    std::vector<int ,aligned_allocator<int >> addr_table(addr_table_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DDR_0(in_bytes / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DDR_1(in_bytes / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DDR_2(in_bytes / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DDR_3(in_bytes / sizeof(int));
    std::vector<int ,aligned_allocator<int>> out(out_bytes);

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
    OCL_CHECK(err, cl::Buffer buffer_in_addr_table   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            addr_table_size, addr_table.data(), &err));

    OCL_CHECK(err, cl::Buffer buffer_in_vec_ID_DDR_0   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            in_bytes, vec_ID_DDR_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_vec_ID_DDR_1   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            in_bytes, vec_ID_DDR_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_vec_ID_DDR_2   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            in_bytes, vec_ID_DDR_2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_vec_ID_DDR_3   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            in_bytes, vec_ID_DDR_3.data(), &err));

	// Allocate Global Memory for sourcce_hw_results
    OCL_CHECK(err, cl::Buffer buffer_out(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            out_bytes, out.data(), &err));

    OCL_CHECK(err, err = krnl_vector_add.setArg(0, buffer_in_addr_table));
    OCL_CHECK(err, err = krnl_vector_add.setArg(1, buffer_in_vec_ID_DDR_0));
    OCL_CHECK(err, err = krnl_vector_add.setArg(2, buffer_in_vec_ID_DDR_1));
    OCL_CHECK(err, err = krnl_vector_add.setArg(3, buffer_in_vec_ID_DDR_2));
    OCL_CHECK(err, err = krnl_vector_add.setArg(4, buffer_in_vec_ID_DDR_3));
    OCL_CHECK(err, err = krnl_vector_add.setArg(5, buffer_out));

    OCL_CHECK(err, err = krnl_vector_add.setArg(6, query_num));
    OCL_CHECK(err, err = krnl_vector_add.setArg(7, nlist));
    OCL_CHECK(err, err = krnl_vector_add.setArg(8, iter_num_per_query_per_ADC_PE));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in_addr_table,
        buffer_in_vec_ID_DDR_0, buffer_in_vec_ID_DDR_1, buffer_in_vec_ID_DDR_2, buffer_in_vec_ID_DDR_3},0/* 0 means from host*/));

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out},CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

// OPENCL HOST CODE AREA END

    std::cout << "TEST FINISHED (NO RESULT CHECK)" << std::endl; 

    return  0;
}
