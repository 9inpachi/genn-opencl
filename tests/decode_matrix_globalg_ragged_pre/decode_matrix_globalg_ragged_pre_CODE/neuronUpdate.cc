#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateNeuronsProgramSrc = R"(typedef float scalar;

__kernel void preNeuronResetKernel(__global unsigned int* d_glbSpkCntPost, __global unsigned int* d_glbSpkCntPre) {
    size_t groupId = get_group_id(0);
    const size_t localId = get_local_id(0);
    unsigned int id = 32 * groupId + localId;
    if(id == 0) {
        d_glbSpkCntPost[0] = 0;
    }
    else if(id == 1) {
        d_glbSpkCntPre[0] = 0;
    }
}

__kernel void updateNeuronsKernel(const float DT, __global unsigned int* d_glbSpkCntPre, __global unsigned int* d_glbSpkPre, __global scalar* d_inSynSyn, __global scalar* d_xPost, float t) {
    size_t groupId = get_group_id(0);
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0); 
    volatile __local unsigned int shSpk[32];
    volatile __local unsigned int shPosSpk;
    volatile __local unsigned int shSpkCount;
    if (localId == 0); {
        shSpkCount = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    // Post
    if(id < 32) {
        
        if(id < 4) {
            scalar lx = d_xPost[id];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynSyn = d_inSynSyn[id];
            Isyn += linSynSyn; linSynSyn = 0;
            // calculate membrane potential
            lx= Isyn;
            
            d_xPost[id] = lx;
            // the post-synaptic dynamics
            
            d_inSynSyn[id] = linSynSyn;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Pre
    if(id >= 32 && id < 64) {
        const unsigned int lid = id - 32;
        
        if(lid < 10) {
            
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            // test for and register a true spike
            if (0) {
                const unsigned int spkIdx = atomic_add(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomic_add(&d_glbSpkCntPre[0], shSpkCount);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < shSpkCount) {
            const unsigned int n = shSpk[localId];
            d_glbSpkPre[shPosSpk + localId] = n;
        }
    }
    
}
)";

// Initialize the neuronUpdate kernels
void updateNeuronsProgramKernels() {
    preNeuronResetKernel = cl::Kernel(updateNeuronsProgram, "preNeuronResetKernel");
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(0, d_glbSpkCntPost));
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(1, d_glbSpkCntPre));
    
    updateNeuronsKernel = cl::Kernel(updateNeuronsProgram, "updateNeuronsKernel");
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(0, DT));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(1, d_glbSpkCntPre));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(2, d_glbSpkPre));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(3, d_inSynSyn));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(4, d_xPost));
}

void updateNeurons(float) {
     {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, cl::NDRange(32)));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
     {
        CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(5, t));
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, cl::NDRange(32)));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
