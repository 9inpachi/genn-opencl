#include "definitionsInternal.h"


extern "C" const char* initProgramSrc = R"(typedef float scalar;

__kernel void initializeKernel(__global unsigned int* d_glbSpkCntpost, __global unsigned int* d_glbSpkCntpre, __global unsigned int* d_glbSpkpost, __global unsigned int* d_glbSpkpre, __global scalar* d_inSynsyn, __global unsigned int* d_indsyn, __global unsigned int* d_rowLengthsyn, __global scalar* d_ssyn, unsigned int deviceRNGSeed) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    // ------------------------------------------------------------------------
    // Local neuron groups
    // post
    if(id < 32) {
        // only do this for existing neurons
        if(id < 10) {
            if(id == 0) {
                for (unsigned int d = 0; d < 21; d++) {
                    d_glbSpkCntpost[d] = 0;
                }
            }
            for (unsigned int d = 0; d < 21; d++) {
                d_glbSpkpost[(d * 10) + id] = 0;
            }
            d_inSynsyn[id] = 0.000000f;
             {
                scalar initVal;
                initVal = (-3.40282346638528860e+38f);
                for (unsigned int d = 0; d < 21; d++) {
                    d_ssyn[(d * 10) + id] = initVal;
                }
            }
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
            // current source variables
        }
    }
    
    
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
    // syn
    if(id >= 64 && id < 96) {
        const unsigned int lid = id - 64;
        // only do this for existing presynaptic neurons
        if(lid < 10) {
            d_rowLengthsyn[lid] = 0;
            // Build sparse connectivity
            while(true) {
                d_indsyn[(lid * 1) + (d_rowLengthsyn[lid]++)] = lid;
                break;
                
            }
        }
    }
    
}

__kernel void initializeSparseKernel(__global unsigned int* d_colLengthsyn, __global unsigned int* d_indsyn, __global unsigned int* d_remapsyn, __global unsigned int* d_rowLengthsyn, __global scalar* d_wsyn) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    __local unsigned int shRowLength[32];
    // syn
    if(id < 32) {
        unsigned int idx = id;
        for(unsigned int r = 0; r < 1; r++) {
            const unsigned numRowsInBlock = (r == 0) ? 10 : 32;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (localId < numRowsInBlock) {
                shRowLength[localId] = d_rowLengthsyn[(r * 32) + localId];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for(unsigned int i = 0; i < numRowsInBlock; i++) {
                if(id < shRowLength[i]) {
                     {
                        d_wsyn[(((r * 32) + i) * 1) + id] = (0.00000000000000000e+00f);
                    }
                     {
                        const unsigned int postIndex = d_indsyn[idx];
                        const unsigned int colLocation = atomic_add(&d_colLengthsyn[postIndex], 1);
                        const unsigned int colMajorIndex = (postIndex * 1) + colLocation;
                        d_remapsyn[colMajorIndex] = idx;
                    }
                }
                idx += 1;
            }
        }
    }
    
    
}

)";

// Initialize the initialization kernel(s)
void initProgramKernels() {
    initializeKernel = cl::Kernel(initProgram, "initializeKernel");
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(0, d_glbSpkCntpost));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(1, d_glbSpkCntpre));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(2, d_glbSpkpost));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(3, d_glbSpkpre));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(4, d_inSynsyn));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(5, d_indsyn));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(6, d_rowLengthsyn));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(7, d_ssyn));
    
    initializeSparseKernel = cl::Kernel(initProgram, "initializeSparseKernel");
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(0, d_colLengthsyn));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(1, d_indsyn));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(2, d_remapsyn));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(3, d_rowLengthsyn));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(4, d_wsyn));
}

void initialize() {
     {
        unsigned int deviceRNGSeed = 0;
        
        CHECK_OPENCL_ERRORS(initializeKernel.setArg(8, deviceRNGSeed));
        
        const cl::NDRange globalWorkSize(96, 1);
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
        const cl::NDRange globalWorkSize(32, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(initializeSparseKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
