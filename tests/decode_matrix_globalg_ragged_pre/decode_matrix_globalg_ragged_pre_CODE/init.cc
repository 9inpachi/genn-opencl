#include "definitionsInternal.h"


extern "C" const char* initProgramSrc = R"(typedef float scalar;

__kernel void initializeKernel(__global scalar* d_glbSpkCntPost, __global scalar* d_glbSpkCntPre, __global scalar* d_glbSpkPost, __global scalar* d_glbSpkPre, __global scalar* d_inSynSyn, __global scalar* d_xPost, unsigned int deviceRNGSeed) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    // ------------------------------------------------------------------------
    // Local neuron groups
    // Post
    if(id < 32) {
        // only do this for existing neurons
        if(id < 4) {
            if(id == 0) {
                d_glbSpkCntPost[0] = 0;
            }
            d_glbSpkPost[id] = 0;
             {
                d_xPost[id] = (0.00000000000000000e+00f);
            }
            d_inSynSyn[id] = 0.000000f;
            // current source variables
        }
    }
    
    // Pre
    if(id >= 32 && id < 64) {
        const unsigned int lid = id - 32;
        // only do this for existing neurons
        if(lid < 10) {
            if(lid == 0) {
                d_glbSpkCntPre[0] = 0;
            }
            d_glbSpkPre[lid] = 0;
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
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(0, d_glbSpkCntPost));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(1, d_glbSpkCntPre));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(2, d_glbSpkPost));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(3, d_glbSpkPre));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(4, d_inSynSyn));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(5, d_xPost));
}

void initialize() {
     {
        unsigned int deviceRNGSeed = 0;
        
        CHECK_OPENCL_ERRORS(initializeKernel.setArg(6, deviceRNGSeed));
        
        const cl::NDRange globalWorkSize(64, 1);
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
