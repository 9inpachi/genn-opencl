#include "definitionsInternal.h"


extern "C" const char* initProgramSrc = R"(typedef float scalar;

__kernel void initializeKernel(__global scalar* d_RefracTimeE, __global scalar* d_RefracTimeI, __global scalar* d_VE, __global scalar* d_VI, __global scalar* d_gIE, __global scalar* d_glbSpkCntE, __global scalar* d_glbSpkCntI, __global scalar* d_glbSpkE, __global scalar* d_glbSpkI, __global scalar* d_inSynEE, __global scalar* d_inSynEI, __global scalar* d_inSynIE, __global scalar* d_inSynII, __global unsigned int* d_indEE, __global unsigned int* d_indEI, __global unsigned int* d_indIE, __global unsigned int* d_indII, __global unsigned int* d_rowLengthEE, __global unsigned int* d_rowLengthEI, __global unsigned int* d_rowLengthIE, __global unsigned int* d_rowLengthII, __global scalar* d_sTE, __global scalar* d_sTI, unsigned int deviceRNGSeed) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    // ------------------------------------------------------------------------
    // Local neuron groups
    // E
    if(id < 2016) {
        // only do this for existing neurons
        if(id < 2000) {
            if(id == 0) {
                d_glbSpkCntE[0] = 0;
            }
            d_glbSpkE[id] = 0;
            d_sTE[id] = -TIME_MAX;
             {
                const scalar scale = (-5.00000000000000000e+01f) - (-6.00000000000000000e+01f);
                d_VE[id] = (-6.00000000000000000e+01f) + (uniform($(rng)) * scale);
            }
             {
                d_RefracTimeE[id] = (0.00000000000000000e+00f);
            }
            d_inSynIE[id] = 0.000000f;
            d_inSynEE[id] = 0.000000f;
            // current source variables
        }
    }
    
    // I
    if(id >= 2016 && id < 2528) {
        const unsigned int lid = id - 2016;
        // only do this for existing neurons
        if(lid < 500) {
            if(lid == 0) {
                d_glbSpkCntI[0] = 0;
            }
            d_glbSpkI[lid] = 0;
            d_sTI[lid] = -TIME_MAX;
             {
                const scalar scale = (-5.00000000000000000e+01f) - (-6.00000000000000000e+01f);
                d_VI[lid] = (-6.00000000000000000e+01f) + (uniform($(rng)) * scale);
            }
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
    // EE
    if(id >= 2528 && id < 4544) {
        const unsigned int lid = id - 2528;
        // only do this for existing presynaptic neurons
        if(lid < 2000) {
            d_rowLengthEE[lid] = 0;
            // Build sparse connectivity
            int prevJ = -1;
            while(true) {
                const scalar u = uniform($(rng));
                prevJ += (1 + (int)(logf(u) * (-4.94983164525091084e+01f)));
                if(prevJ < 2000) {
                   d_indEE[(lid * 77) + (d_rowLengthEE[lid]++)] = prevJ;
                }
                else {
                   break;
                }
                
            }
        }
    }
    
    // EI
    if(id >= 4544 && id < 6560) {
        const unsigned int lid = id - 4544;
        // only do this for existing presynaptic neurons
        if(lid < 2000) {
            d_rowLengthEI[lid] = 0;
            // Build sparse connectivity
            int prevJ = -1;
            while(true) {
                const scalar u = uniform($(rng));
                prevJ += (1 + (int)(logf(u) * (-4.94983164525091084e+01f)));
                if(prevJ < 500) {
                   d_indEI[(lid * 31) + (d_rowLengthEI[lid]++)] = prevJ;
                }
                else {
                   break;
                }
                
            }
        }
    }
    
    // IE
    if(id >= 6560 && id < 7072) {
        const unsigned int lid = id - 6560;
        // only do this for existing presynaptic neurons
        if(lid < 500) {
            d_rowLengthIE[lid] = 0;
            // Build sparse connectivity
            int prevJ = -1;
            while(true) {
                const scalar u = uniform($(rng));
                prevJ += (1 + (int)(logf(u) * (-4.94983164525091084e+01f)));
                if(prevJ < 2000) {
                   d_indIE[(lid * 75) + (d_rowLengthIE[lid]++)] = prevJ;
                }
                else {
                   break;
                }
                
            }
        }
    }
    
    // II
    if(id >= 7072 && id < 7584) {
        const unsigned int lid = id - 7072;
        // only do this for existing presynaptic neurons
        if(lid < 500) {
            d_rowLengthII[lid] = 0;
            // Build sparse connectivity
            int prevJ = -1;
            while(true) {
                const scalar u = uniform($(rng));
                prevJ += (1 + (int)(logf(u) * (-4.94983164525091084e+01f)));
                if(prevJ < 500) {
                   d_indII[(lid * 29) + (d_rowLengthII[lid]++)] = prevJ;
                }
                else {
                   break;
                }
                
            }
        }
    }
    
}

__kernel void initializeSparseKernel(__global unsigned int* d_colLengthIE, __global scalar* d_gIE, __global unsigned int* d_indIE, __global unsigned int* d_remapIE, __global unsigned int* d_rowLengthIE) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    __local unsigned int shRowLength[32];
    // IE
    if(id < 96) {
        unsigned int idx = id;
        for(unsigned int r = 0; r < 16; r++) {
            const unsigned numRowsInBlock = (r == 15) ? 20 : 32;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (localId < numRowsInBlock) {
                shRowLength[localId] = d_rowLengthIE[(r * 32) + localId];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for(unsigned int i = 0; i < numRowsInBlock; i++) {
                if(id < shRowLength[i]) {
                     {
                        d_gIE[(((r * 32) + i) * 75) + id] = (0.00000000000000000e+00f);
                    }
                     {
                        const unsigned int postIndex = d_indIE[idx];
                        const unsigned int colLocation = atomic_add(&d_colLengthIE[postIndex], 1);
                        const unsigned int colMajorIndex = (postIndex * 31) + colLocation;
                        d_remapIE[colMajorIndex] = idx;
                    }
                }
                idx += 75;
            }
        }
    }
    
    
}

)";

// Initialize the initialization kernel(s)
void initProgramKernels() {
    initializeKernel = cl::Kernel(initProgram, "initializeKernel");
    std::string err = opencl::clGetErrorString(initializeKernel.setArg(0, d_RefracTimeE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(1, d_RefracTimeI));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(2, d_VE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(3, d_VI));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(4, d_gIE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(5, d_glbSpkCntE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(6, d_glbSpkCntI));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(7, d_glbSpkE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(8, d_glbSpkI));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(9, d_inSynEE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(10, d_inSynEI));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(11, d_inSynIE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(12, d_inSynII));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(13, d_indEE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(14, d_indEI));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(15, d_indIE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(16, d_indII));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(17, d_rowLengthEE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(18, d_rowLengthEI));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(19, d_rowLengthIE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(20, d_rowLengthII));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(21, d_sTE));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(22, d_sTI));
    
    initializeSparseKernel = cl::Kernel(initProgram, "initializeSparseKernel");
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(0, d_colLengthIE));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(1, d_gIE));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(2, d_indIE));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(3, d_remapIE));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(4, d_rowLengthIE));
}

void initialize() {
     {
        unsigned int deviceRNGSeed = 0;
        
        CHECK_OPENCL_ERRORS(initializeKernel.setArg(23, deviceRNGSeed));
        
        const cl::NDRange globalWorkSize(7584, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(initializeKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}

// Initialize all OpenCL elements
void initializeSparse() {
    copyStateToDevice(true);
    copyConnectivityToDevice(true);
     {
        const cl::NDRange globalWorkSize(96, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(initializeSparseKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
