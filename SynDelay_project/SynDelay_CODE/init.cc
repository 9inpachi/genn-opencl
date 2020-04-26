#include "definitionsInternal.h"


extern "C" const char* initProgramSrc = R"(typedef float scalar;

__kernel void initializeKernel(__global scalar* d_UInput, __global scalar* d_UInter, __global scalar* d_UOutput, __global scalar* d_VInput, __global scalar* d_VInter, __global scalar* d_VOutput, __global scalar* d_glbSpkCntInput, __global scalar* d_glbSpkCntInter, __global scalar* d_glbSpkCntOutput, __global scalar* d_glbSpkInput, __global scalar* d_glbSpkInter, __global scalar* d_glbSpkOutput, __global scalar* d_inSynInputInter, __global scalar* d_inSynInputOutput, __global scalar* d_inSynInterOutput, unsigned int deviceRNGSeed) {
    size_t groupId = get_group_id(0);
    size_t localId = get_local_id(0);
    const unsigned int id = 32 * groupId + localId;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // Input
    if(id < 512) {
        // only do this for existing neurons
        if(id < 500) {
            if(id == 0) {
                for (unsigned int d = 0; d < 7; d++) {
                    d_glbSpkCntInput[d] = 0;
                }
            }
            for (unsigned int d = 0; d < 7; d++) {
                d_glbSpkInput[(d * 500) + id] = 0;
            }
             {
                d_VInput[id] = (-6.50000000000000000e+01f);
            }
             {
                d_UInput[id] = (-2.00000000000000000e+01f);
            }
            // current source variables
        }
    }
    
    // Inter
    if(id >= 512 && id < 1024) {
        const unsigned int lid = id - 512;
        // only do this for existing neurons
        if(lid < 500) {
            if(lid == 0) {
                d_glbSpkCntInter[0] = 0;
            }
            d_glbSpkInter[lid] = 0;
             {
                d_VInter[lid] = (-6.50000000000000000e+01f);
            }
             {
                d_UInter[lid] = (-2.00000000000000000e+01f);
            }
            d_inSynInputInter[lid] = 0.000000f;
            // current source variables
        }
    }
    
    // Output
    if(id >= 1024 && id < 1536) {
        const unsigned int lid = id - 1024;
        // only do this for existing neurons
        if(lid < 500) {
            if(lid == 0) {
                d_glbSpkCntOutput[0] = 0;
            }
            d_glbSpkOutput[lid] = 0;
             {
                d_VOutput[lid] = (-6.50000000000000000e+01f);
            }
             {
                d_UOutput[lid] = (-2.00000000000000000e+01f);
            }
            d_inSynInterOutput[lid] = 0.000000f;
            d_inSynInputOutput[lid] = 0.000000f;
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
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(0, d_UInput));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(1, d_UInter));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(2, d_UOutput));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(3, d_VInput));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(4, d_VInter));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(5, d_VOutput));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(6, d_glbSpkCntInput));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(7, d_glbSpkCntInter));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(8, d_glbSpkCntOutput));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(9, d_glbSpkInput));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(10, d_glbSpkInter));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(11, d_glbSpkOutput));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(12, d_inSynInputInter));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(13, d_inSynInputOutput));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(14, d_inSynInterOutput));
}

void initialize() {
    unsigned int deviceRNGSeed = 0;
    
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(15, deviceRNGSeed));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(initializeKernel, cl::NullRange, cl::NDRange(32)));
    CHECK_OPENCL_ERRORS(commandQueue.finish());
}

// Initialize all OpenCL elements
void initializeSparse() {
    copyStateToDevice(true);
}
