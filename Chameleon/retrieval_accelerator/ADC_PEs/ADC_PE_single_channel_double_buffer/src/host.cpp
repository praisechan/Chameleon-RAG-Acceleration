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

    int query_num = 10000;
    int nprobe = 32;
    int compute_iter_per_PE = 1000; // per nprobe

    size_t in_bytes = compute_iter_per_PE * 64; 
    size_t out_bytes = 128;
    std::vector<int ,aligned_allocator<int >> in(in_bytes / sizeof(int));
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
    OCL_CHECK(err, cl::Buffer buffer_in   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            in_bytes, in.data(), &err));

	// Allocate Global Memory for sourcce_hw_results
    OCL_CHECK(err, cl::Buffer buffer_out(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            out_bytes, out.data(), &err));

    OCL_CHECK(err, err = krnl_vector_add.setArg(0, buffer_in));
    OCL_CHECK(err, err = krnl_vector_add.setArg(1, buffer_out));
    OCL_CHECK(err, err = krnl_vector_add.setArg(2, query_num));
    OCL_CHECK(err, err = krnl_vector_add.setArg(3, nprobe));
    OCL_CHECK(err, err = krnl_vector_add.setArg(4, compute_iter_per_PE));
    
    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in},0/* 0 means from host*/));

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out},CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

// OPENCL HOST CODE AREA END

    std::cout << "TEST FINISHED (NO RESULT CHECK)" << std::endl; 

    return  0;
}
