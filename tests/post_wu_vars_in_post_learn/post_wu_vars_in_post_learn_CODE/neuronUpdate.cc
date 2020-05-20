#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateNeuronsProgramSrc = R"(typedef float scalar;

__kernel void preNeuronResetKernel(__global unsigned int* d_glbSpkCntpost, __global unsigned int* d_glbSpkCntpre, volatile unsigned int spkQuePtrpost) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    if(id == 0) {
        d_glbSpkCntpost[spkQuePtrpost] = 0;
    }
    else if(id == 1) {
        d_glbSpkCntpre[0] = 0;
    }
}

__kernel void updateNeuronsKernel(const float DT, __global unsigned int* d_glbSpkCntpost, __global unsigned int* d_glbSpkCntpre, __global unsigned int* d_glbSpkpost, __global unsigned int* d_glbSpkpre, __global scalar* d_inSynsyn, __global scalar* d_ssyn, volatile unsigned int spkQuePtrpost, float t) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    volatile __local unsigned int shSpk[32];
    volatile __local unsigned int shPosSpk;
    volatile __local unsigned int shSpkCount;
    if (localId == 0); {
        shSpkCount = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    // post
    if(id < 32) {
        const unsigned int readDelayOffset = (((spkQuePtrpost + 20) % 21) * 10);
        const unsigned int writeDelayOffset = (spkQuePtrpost * 10);
        
        if(id < 10) {
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn = d_inSynsyn[id];
            Isyn += linSynsyn; linSynsyn = 0;
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            // test for and register a true spike
            if (t >= (scalar)id && fmod(t - (scalar)id, 10.0f)< 1e-4f) {
                const unsigned int spkIdx = atomic_add(&shSpkCount, 1);
                shSpk[spkIdx] = id;
            }
            else {
                d_ssyn[writeDelayOffset + id] = d_ssyn[readDelayOffset + id];
            }
            // the post-synaptic dynamics
            
            d_inSynsyn[id] = linSynsyn;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomic_add(&d_glbSpkCntpost[spkQuePtrpost], shSpkCount);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < shSpkCount) {
            const unsigned int n = shSpk[localId];
             {
                // perform postsynaptic update required for syn
                scalar ls = d_ssyn[readDelayOffset + n];
                ls = t;
                d_ssyn[writeDelayOffset + n] = ls;
            }
            d_glbSpkpost[writeDelayOffset + shPosSpk + localId] = n;
        }
    }
    
    // pre
    if(id >= 32 && id < 64) {
        const unsigned int lid = id - 32;
        
        if(lid < 10) {
            
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            // test for and register a true spike
            if (true) {
                const unsigned int spkIdx = atomic_add(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomic_add(&d_glbSpkCntpre[0], shSpkCount);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < shSpkCount) {
            const unsigned int n = shSpk[localId];
            d_glbSpkpre[shPosSpk + localId] = n;
        }
    }
    
}
)";

// Initialize the neuronUpdate kernels
void updateNeuronsProgramKernels() {
    preNeuronResetKernel = cl::Kernel(updateNeuronsProgram, "preNeuronResetKernel");
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(0, d_glbSpkCntpost));
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(1, d_glbSpkCntpre));
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(2, spkQuePtrpost));
    
    updateNeuronsKernel = cl::Kernel(updateNeuronsProgram, "updateNeuronsKernel");
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(0, DT));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(1, d_glbSpkCntpost));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(2, d_glbSpkCntpre));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(3, d_glbSpkpost));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(4, d_glbSpkpre));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(5, d_inSynsyn));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(6, d_ssyn));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(7, spkQuePtrpost));
}

void updateNeurons(float t) {
     {
        CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(2, spkQuePtrpost));
        const cl::NDRange globalWorkSize(32, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
     {
        CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(7, spkQuePtrpost));
        CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(8, t));
        
        const cl::NDRange globalWorkSize(64, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
