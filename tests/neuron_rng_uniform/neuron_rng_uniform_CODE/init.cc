#include "definitionsInternal.h"


extern "C" const char* initProgramSrc = R"(typedef float scalar;

#define fmodf fmod
#define TIME_MIN 1.175494351e-38f
#define TIME_MAX 3.402823466e+38f

__kernel void initializeKernel(__global unsigned int* d_glbSpkCntPop, __global unsigned int* d_glbSpkPop, __global scalar* d_xPop, unsigned int deviceRNGSeed) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    // ------------------------------------------------------------------------
    // Local neuron groups
    // Pop
    if(id < 1024) {
        // only do this for existing neurons
        if(id < 1000) {
            if(id == 0) {
                d_glbSpkCntPop[0] = 0;
            }
            d_glbSpkPop[id] = 0;
             {
                d_xPop[id] = (0.00000000000000000e+00f);
            }
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
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(0, d_glbSpkCntPop));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(1, d_glbSpkPop));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(2, d_xPop));
}

void initialize() {
     {
        unsigned int deviceRNGSeed = 0;
        
        CHECK_OPENCL_ERRORS(initializeKernel.setArg(3, deviceRNGSeed));
        
        const cl::NDRange globalWorkSize(1024, 1);
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
