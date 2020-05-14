#include "definitionsInternal.h"


extern "C" const char* initProgramSrc = R"(typedef float scalar;

__kernel void initializeKernel(__global scalar* d_RefracTimeE, __global scalar* d_RefracTimeI, __global scalar* d_VE, __global scalar* d_VI, __global scalar* d_glbSpkCntE, __global scalar* d_glbSpkCntI, __global scalar* d_glbSpkE, __global scalar* d_glbSpkI, __global scalar* d_inSynEE, __global scalar* d_inSynEI, __global scalar* d_inSynIE, __global scalar* d_inSynII, unsigned int deviceRNGSeed) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    // ------------------------------------------------------------------------
    // Local neuron groups
    // E
    if(id < 40000) {
        // only do this for existing neurons
        if(id < 40000) {
            if(id == 0) {
                d_glbSpkCntE[0] = 0;
            }
            d_glbSpkE[id] = 0;
             {
                d_RefracTimeE[id] = (0.00000000000000000e+00f);
            }
            d_inSynIE[id] = 0.000000f;
            d_inSynEE[id] = 0.000000f;
            // current source variables
        }
    }
    
    // I
    if(id >= 40000 && id < 50016) {
        const unsigned int lid = id - 40000;
        // only do this for existing neurons
        if(lid < 10000) {
            if(lid == 0) {
                d_glbSpkCntI[0] = 0;
            }
            d_glbSpkI[lid] = 0;
             {
                d_RefracTimeI[lid] = (0.00000000000000000e+00f);
            }
            d_inSynII[lid] = 0.000000f;
            d_inSynEI[lid] = 0.000000f;
            // current source variables
        }
    }
    
    
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
}

)";

// Initialize the initialization kernel(s)
void initProgramKernels() {
    initializeKernel = cl::Kernel(initProgram, "initializeKernel");
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(0, d_RefracTimeE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(1, d_RefracTimeI));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(2, d_VE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(3, d_VI));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(4, d_glbSpkCntE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(5, d_glbSpkCntI));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(6, d_glbSpkE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(7, d_glbSpkI));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(8, d_inSynEE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(9, d_inSynEI));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(10, d_inSynIE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(11, d_inSynII));
}

void initialize() {
     {
        unsigned int deviceRNGSeed = 0;
        
        CHECK_OPENCL_ERRORS(initializeKernel.setArg(12, deviceRNGSeed));
        
        const cl::NDRange globalWorkSize(50016, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(initializeKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}

// Initialize all OpenCL elements
void initializeSparse() {
    copyStateToDevice(true);
    copyConnectivityToDevice(true);
}
