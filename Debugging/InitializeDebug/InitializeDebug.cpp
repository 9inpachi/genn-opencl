#include <iostream>
#include <stdexcept>
// OpenCL includes
#define CL_HPP_TARGET_OPENCL_VERSION 120 
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "CL/cl2.hpp"
// ------------------------------------------------------------------------
// Macros
// ------------------------------------------------------------------------
// Helper macro for error-checking OpenCL calls
#define CHECK_OPENCL_ERRORS(call) {\
    cl_int error = call;\
    if (error != CL_SUCCESS) {\
        throw std::runtime_error(__FILE__": " + std::to_string(__LINE__) + ": opencl error " + std::to_string(error)); \
    }\
}
#define CHECK_OPENCL_ERRORS_POINTER(call) {\
    cl_int error;\
    call;\
    if (error != CL_SUCCESS) {\
        throw std::runtime_error(__FILE__": " + std::to_string(__LINE__) + ": opencl error " + std::to_string(error));\
    }\
}
// ------------------------------------------------------------------------
// Globals
// ------------------------------------------------------------------------
// OpenCL variables
cl::Context clContext;
cl::Device clDevice;
cl::CommandQueue commandQueue;
// Buffers and pointers
float* xPost;
cl::Buffer h_xPost;
cl::Buffer d_xPost;
cl::Buffer d_mergedNeuronInitGroup0;
// Programs and kernels
cl::Program initializeProgram;
cl::Kernel initializeKernel;
cl::Kernel buildNeuronInit0Kernel;
// ------------------------------------------------------------------------
// Program source
// ------------------------------------------------------------------------
const char* initializeSrc = R"(
struct MergedNeuronInitGroup0
 {
    unsigned int numNeurons;
    __global float* x;   
};
__kernel void buildNeuronInit0Kernel(__global struct MergedNeuronInitGroup0 *group, unsigned int idx, unsigned int numNeurons, __global float* x) {
    group[idx].numNeurons = numNeurons;
    group[idx].x = x;
}
__kernel void initializeKernel(__global struct MergedNeuronInitGroup0 *d_mergedNeuronInitGroup0) {
    const unsigned int id = get_global_id(0);
    if(id < 32) {
        __global struct MergedNeuronInitGroup0 *group = &d_mergedNeuronInitGroup0[0]; 
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            group->x[lid] = (0.00000000000000000e+00f);
            
            //**UNCOMMENT AND IT MAGICALLY WORKS ON AMD**
            printf("AMD!\n");
        }
    }
})";
// ------------------------------------------------------------------------
// Functions
// ------------------------------------------------------------------------
void allocateMem() {
    // Get platforms
    std::vector<cl::Platform> platforms; 
    cl::Platform::get(&platforms);
    // Get platform devices
    // **CHANGE PLATFORM HERE**
    std::vector<cl::Device> platformDevices; 
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
    // Select device and create context and command queue
    // **CHANGE DEVICE INDEX HERE**
    clDevice = platformDevices[0];
    clContext = cl::Context(clDevice);
    std::cout << clDevice.getInfo<CL_DEVICE_NAME>() << std::endl;
    commandQueue = cl::CommandQueue(clContext, clDevice, 0);
    // Allocate buffers 
    // **HACK** wildly over-allocate
    const cl_int deviceAddressBytes = clDevice.getInfo<CL_DEVICE_ADDRESS_BITS>() / 8;
    const size_t structSize = (deviceAddressBytes + sizeof(uint32_t)) * 2;
    CHECK_OPENCL_ERRORS_POINTER(d_mergedNeuronInitGroup0 = cl::Buffer(clContext, CL_MEM_READ_WRITE, structSize, nullptr, &error));
    CHECK_OPENCL_ERRORS_POINTER(d_xPost = cl::Buffer(clContext, CL_MEM_READ_WRITE, 4 * sizeof(float), nullptr, &error));
    CHECK_OPENCL_ERRORS_POINTER(h_xPost = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 4 * sizeof(float), nullptr, &error));
    CHECK_OPENCL_ERRORS_POINTER(xPost = (float*)commandQueue.enqueueMapBuffer(h_xPost, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 4 * sizeof(float), nullptr, nullptr, &error));
    
    // Build OpenCL program
    CHECK_OPENCL_ERRORS_POINTER(initializeProgram = cl::Program(clContext, initializeSrc, false, &error));
    if(initializeProgram.build("-cl-std=CL1.2 -I decode_matrix_globalg_dense_CODE/opencl/clRNG/include -I opencl/clRNG/include") != CL_SUCCESS) {
        std::cerr << initializeProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clDevice);
        throw std::runtime_error("Initialize program compile error");
    }
    
    // Configure merged struct building kernels
    CHECK_OPENCL_ERRORS_POINTER(buildNeuronInit0Kernel = cl::Kernel(initializeProgram, "buildNeuronInit0Kernel", &error));
    // Launch kernel to setup merge group structure
    CHECK_OPENCL_ERRORS(buildNeuronInit0Kernel.setArg(0, d_mergedNeuronInitGroup0));
    CHECK_OPENCL_ERRORS(buildNeuronInit0Kernel.setArg(1, 0));
    CHECK_OPENCL_ERRORS(buildNeuronInit0Kernel.setArg(2, 4));
    CHECK_OPENCL_ERRORS(buildNeuronInit0Kernel.setArg(3, d_xPost));
    const cl::NDRange globalWorkSize(1, 1);
    const cl::NDRange localWorkSize(1, 1);
    CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(buildNeuronInit0Kernel, cl::NullRange, globalWorkSize, localWorkSize));
    // Configure initialization kernel
    CHECK_OPENCL_ERRORS_POINTER(initializeKernel = cl::Kernel(initializeProgram, "initializeKernel", &error));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(0, d_mergedNeuronInitGroup0));   
}
int main()
{
    allocateMem();
    // Launch initialize kernel
    const cl::NDRange globalWorkSize(32, 1);
    const cl::NDRange localWorkSize(32, 1);
    CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(initializeKernel, cl::NullRange, globalWorkSize, localWorkSize));
    // Read buffer
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_xPost, CL_TRUE, 0, 4 * sizeof(float), xPost));
    for(unsigned int j = 0; j < 4; j++) {
        std::cout << xPost[j] << ",";
    }
    std::cout << std::endl;
    return EXIT_SUCCESS;
}
