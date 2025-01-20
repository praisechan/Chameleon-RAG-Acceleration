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


    size_t query_num = 1000;
    size_t nlist = 262144; 
    size_t nprobe = 32;

    size_t entries_per_cell = 100; // per nprobe

    size_t nlist_PQ_codes_start_addr_bytes = nlist * sizeof(int);
    size_t nlist_num_vecs_bytes = nlist * sizeof(int);
    // size_t cell_ID_DRAM_bytes = query_num * nprobe * sizeof(int);
    // size_t LUT_DRAM_bytes = query_num * nprobe * LUT_ENTRY_NUM * M * sizeof(float);
    size_t PQ_codes_DRAM_0_bytes = nlist * entries_per_cell * 64;
    size_t PQ_codes_DRAM_1_bytes = nlist * entries_per_cell * 64;
    size_t PQ_codes_DRAM_2_bytes = nlist * entries_per_cell * 64;
    size_t PQ_codes_DRAM_3_bytes = nlist * entries_per_cell * 64;

    std::cout << "Allocating memory...\n";
//     std::cout << "size in GB\n" << nlist_PQ_codes_start_addr_bytes / 1e9 << "\n" << 
// 	    nlist_num_vecs_bytes / 1e9 << "\n" <<
// 	    cell_ID_DRAM_bytes / 1e9 << "\n" <<
// 	    LUT_DRAM_bytes / 1e9 << "\n" << 
// 	    PQ_codes_DRAM_0_bytes / 1e9 << "\n";

    std::vector<int ,aligned_allocator<int>> nlist_PQ_codes_start_addr(nlist_PQ_codes_start_addr_bytes / long(sizeof(int)));
    std::vector<int ,aligned_allocator<int>> nlist_num_vecs(nlist_num_vecs_bytes / long(sizeof(int)));
    // std::vector<int ,aligned_allocator<int>> cell_ID_DRAM(cell_ID_DRAM_bytes / long(sizeof(int)));
    // std::vector<int ,aligned_allocator<int>> LUT_DRAM(LUT_DRAM_bytes / long(sizeof(int)));
    std::vector<int ,aligned_allocator<int>> PQ_codes_DRAM_0(PQ_codes_DRAM_0_bytes / long(sizeof(int)));
    std::vector<int ,aligned_allocator<int>> PQ_codes_DRAM_1(PQ_codes_DRAM_1_bytes / long(sizeof(int)));
    std::vector<int ,aligned_allocator<int>> PQ_codes_DRAM_2(PQ_codes_DRAM_2_bytes / long(sizeof(int)));
    std::vector<int ,aligned_allocator<int>> PQ_codes_DRAM_3(PQ_codes_DRAM_3_bytes / long(sizeof(int)));

    // init compute iter per PE
    int compute_iter_per_cell = 1000;
    for (int i = 0; i < nlist; i++) {
	nlist_num_vecs[i] = compute_iter_per_cell * ADC_PE_NUM;
    }

//     std::cout << "max_size: " << PQ_codes_DRAM_0.max_size() << "\n";
    
    size_t out_bytes = 128;
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
    OCL_CHECK(err, cl::Buffer buffer_nlist_PQ_codes_start_addr   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            nlist_PQ_codes_start_addr_bytes, nlist_PQ_codes_start_addr.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_nlist_num_vecs   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            nlist_num_vecs_bytes, nlist_num_vecs.data(), &err));
    // OCL_CHECK(err, cl::Buffer buffer_cell_ID_DRAM   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
    //         cell_ID_DRAM_bytes, cell_ID_DRAM.data(), &err));
    // OCL_CHECK(err, cl::Buffer buffer_LUT_DRAM   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
    //         LUT_DRAM_bytes, LUT_DRAM.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_PQ_codes_DRAM_0   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            PQ_codes_DRAM_0_bytes, PQ_codes_DRAM_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_PQ_codes_DRAM_1   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            PQ_codes_DRAM_1_bytes, PQ_codes_DRAM_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_PQ_codes_DRAM_2   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            PQ_codes_DRAM_2_bytes, PQ_codes_DRAM_2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_PQ_codes_DRAM_3   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            PQ_codes_DRAM_3_bytes, PQ_codes_DRAM_3.data(), &err));

	// Allocate Global Memory for sourcce_hw_results
    OCL_CHECK(err, cl::Buffer buffer_out(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            out_bytes, out.data(), &err));

    // in init
    OCL_CHECK(err, err = krnl_vector_add.setArg(0, int(query_num)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(1, int(nlist)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(2, int(nprobe)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(3, buffer_nlist_PQ_codes_start_addr));
    OCL_CHECK(err, err = krnl_vector_add.setArg(4, buffer_nlist_num_vecs));
    // in runtime
    // OCL_CHECK(err, err = krnl_vector_add.setArg(5, buffer_cell_ID_DRAM));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(6, buffer_LUT_DRAM));
    OCL_CHECK(err, err = krnl_vector_add.setArg(5, buffer_PQ_codes_DRAM_0));
    OCL_CHECK(err, err = krnl_vector_add.setArg(6, buffer_PQ_codes_DRAM_1));
    OCL_CHECK(err, err = krnl_vector_add.setArg(7, buffer_PQ_codes_DRAM_2));
    OCL_CHECK(err, err = krnl_vector_add.setArg(8, buffer_PQ_codes_DRAM_3));
    // out
    OCL_CHECK(err, err = krnl_vector_add.setArg(9, buffer_out));
    
    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        buffer_nlist_PQ_codes_start_addr,
        buffer_nlist_num_vecs,
        // buffer_cell_ID_DRAM, 
        // buffer_LUT_DRAM, 
        buffer_PQ_codes_DRAM_0,
        buffer_PQ_codes_DRAM_1,
        buffer_PQ_codes_DRAM_2,
        buffer_PQ_codes_DRAM_3,
        },0/* 0 means from host*/));

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out},CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

// OPENCL HOST CODE AREA END

    std::cout << "TEST FINISHED (NO RESULT CHECK)" << std::endl; 

    return  0;
}
