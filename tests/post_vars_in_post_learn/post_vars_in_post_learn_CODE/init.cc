#include "definitionsInternal.h"


extern "C" const char* initProgramSrc = R"(typedef float scalar;

__kernel void initializeKernel(__global scalar* d_glbSpkCntpost, __global scalar* d_glbSpkCntpre, __global scalar* d_glbSpkpost, __global scalar* d_glbSpkpre, __global scalar* d_inSynsyn0, __global scalar* d_inSynsyn1, __global scalar* d_inSynsyn2, __global scalar* d_inSynsyn3, __global scalar* d_inSynsyn4, __global scalar* d_inSynsyn5, __global scalar* d_inSynsyn6, __global scalar* d_inSynsyn7, __global scalar* d_inSynsyn8, __global scalar* d_inSynsyn9, __global scalar* d_shiftpost, __global scalar* d_shiftpre, __global scalar* d_wsyn0, __global scalar* d_wsyn1, __global scalar* d_wsyn2, __global scalar* d_wsyn3, __global scalar* d_wsyn4, __global scalar* d_wsyn5, __global scalar* d_wsyn6, __global scalar* d_wsyn7, __global scalar* d_wsyn8, __global scalar* d_wsyn9, __global scalar* d_xpost, __global scalar* d_xpre, unsigned int deviceRNGSeed) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    // ------------------------------------------------------------------------
    // Local neuron groups
    // post
    if(id < 32) {
        // only do this for existing neurons
        if(id < 10) {
            if(id == 0) {
                d_glbSpkCntpost[0] = 0;
            }
            d_glbSpkpost[id] = 0;
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                for (unsigned int d = 0; d < 1; d++) {
                    d_xpost[(d * 10) + id] = initVal;
                }
            }
            d_inSynsyn9[id] = 0.000000f;
            d_inSynsyn8[id] = 0.000000f;
            d_inSynsyn7[id] = 0.000000f;
            d_inSynsyn6[id] = 0.000000f;
            d_inSynsyn5[id] = 0.000000f;
            d_inSynsyn4[id] = 0.000000f;
            d_inSynsyn3[id] = 0.000000f;
            d_inSynsyn2[id] = 0.000000f;
            d_inSynsyn1[id] = 0.000000f;
            d_inSynsyn0[id] = 0.000000f;
            // current source variables
        }
    }
    
    // pre
    if(id >= 32 && id < 64) {
        const unsigned int lid = id - 32;
        // only do this for existing neurons
        if(lid < 10) {
            if(lid == 0) {
                d_glbSpkCntpre[0] = 0;
            }
            d_glbSpkpre[lid] = 0;
             {
                d_xpre[lid] = (0.00000000000000000e+00f);
            }
            // current source variables
        }
    }
    
    
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    // syn0
    if(id >= 64 && id < 96) {
        const unsigned int lid = id - 64;
        // only do this for existing postsynaptic neurons
        if(lid < 10) {
            for(unsigned int i = 0; i < 10; i++) {
                 {
                    d_wsyn0[(i * 10) + lid] = (0.00000000000000000e+00f);
                }
            }
        }
    }
    
    // syn1
    if(id >= 96 && id < 128) {
        const unsigned int lid = id - 96;
        // only do this for existing postsynaptic neurons
        if(lid < 10) {
            for(unsigned int i = 0; i < 10; i++) {
                 {
                    d_wsyn1[(i * 10) + lid] = (0.00000000000000000e+00f);
                }
            }
        }
    }
    
    // syn2
    if(id >= 128 && id < 160) {
        const unsigned int lid = id - 128;
        // only do this for existing postsynaptic neurons
        if(lid < 10) {
            for(unsigned int i = 0; i < 10; i++) {
                 {
                    d_wsyn2[(i * 10) + lid] = (0.00000000000000000e+00f);
                }
            }
        }
    }
    
    // syn3
    if(id >= 160 && id < 192) {
        const unsigned int lid = id - 160;
        // only do this for existing postsynaptic neurons
        if(lid < 10) {
            for(unsigned int i = 0; i < 10; i++) {
                 {
                    d_wsyn3[(i * 10) + lid] = (0.00000000000000000e+00f);
                }
            }
        }
    }
    
    // syn4
    if(id >= 192 && id < 224) {
        const unsigned int lid = id - 192;
        // only do this for existing postsynaptic neurons
        if(lid < 10) {
            for(unsigned int i = 0; i < 10; i++) {
                 {
                    d_wsyn4[(i * 10) + lid] = (0.00000000000000000e+00f);
                }
            }
        }
    }
    
    // syn5
    if(id >= 224 && id < 256) {
        const unsigned int lid = id - 224;
        // only do this for existing postsynaptic neurons
        if(lid < 10) {
            for(unsigned int i = 0; i < 10; i++) {
                 {
                    d_wsyn5[(i * 10) + lid] = (0.00000000000000000e+00f);
                }
            }
        }
    }
    
    // syn6
    if(id >= 256 && id < 288) {
        const unsigned int lid = id - 256;
        // only do this for existing postsynaptic neurons
        if(lid < 10) {
            for(unsigned int i = 0; i < 10; i++) {
                 {
                    d_wsyn6[(i * 10) + lid] = (0.00000000000000000e+00f);
                }
            }
        }
    }
    
    // syn7
    if(id >= 288 && id < 320) {
        const unsigned int lid = id - 288;
        // only do this for existing postsynaptic neurons
        if(lid < 10) {
            for(unsigned int i = 0; i < 10; i++) {
                 {
                    d_wsyn7[(i * 10) + lid] = (0.00000000000000000e+00f);
                }
            }
        }
    }
    
    // syn8
    if(id >= 320 && id < 352) {
        const unsigned int lid = id - 320;
        // only do this for existing postsynaptic neurons
        if(lid < 10) {
            for(unsigned int i = 0; i < 10; i++) {
                 {
                    d_wsyn8[(i * 10) + lid] = (0.00000000000000000e+00f);
                }
            }
        }
    }
    
    // syn9
    if(id >= 352 && id < 384) {
        const unsigned int lid = id - 352;
        // only do this for existing postsynaptic neurons
        if(lid < 10) {
            for(unsigned int i = 0; i < 10; i++) {
                 {
                    d_wsyn9[(i * 10) + lid] = (0.00000000000000000e+00f);
                }
            }
        }
    }
    
    
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
}

)";

// Initialize the initialization kernel(s)
void initProgramKernels() {
    initializeKernel = cl::Kernel(initProgram, "initializeKernel");
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(0, d_glbSpkCntpost));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(1, d_glbSpkCntpre));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(2, d_glbSpkpost));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(3, d_glbSpkpre));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(4, d_inSynsyn0));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(5, d_inSynsyn1));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(6, d_inSynsyn2));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(7, d_inSynsyn3));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(8, d_inSynsyn4));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(9, d_inSynsyn5));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(10, d_inSynsyn6));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(11, d_inSynsyn7));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(12, d_inSynsyn8));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(13, d_inSynsyn9));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(14, d_shiftpost));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(15, d_shiftpre));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(16, d_wsyn0));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(17, d_wsyn1));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(18, d_wsyn2));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(19, d_wsyn3));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(20, d_wsyn4));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(21, d_wsyn5));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(22, d_wsyn6));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(23, d_wsyn7));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(24, d_wsyn8));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(25, d_wsyn9));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(26, d_xpost));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(27, d_xpre));
}

void initialize() {
     {
        unsigned int deviceRNGSeed = 0;
        
        CHECK_OPENCL_ERRORS(initializeKernel.setArg(28, deviceRNGSeed));
        
        const cl::NDRange globalWorkSize(384, 1);
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
