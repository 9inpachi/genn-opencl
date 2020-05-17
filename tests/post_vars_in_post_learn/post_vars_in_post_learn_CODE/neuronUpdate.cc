#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateNeuronsProgramSrc = R"(typedef float scalar;

__kernel void preNeuronResetKernel(__global unsigned int* d_glbSpkCntpost, __global unsigned int* d_glbSpkCntpre, volatile unsigned int spkQuePtrpre) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    if(id == 0) {
        d_glbSpkCntpost[0] = 0;
    }
    else if(id == 1) {
        spkQuePtrpre = (spkQuePtrpre + 1) % 10;
        d_glbSpkCntpre[0] = 0;
    }
}

__kernel void updateNeuronsKernel(const float DT, __global unsigned int* d_glbSpkCntpost, __global unsigned int* d_glbSpkCntpre, __global unsigned int* d_glbSpkpost, __global unsigned int* d_glbSpkpre, __global scalar* d_inSynsyn0, __global scalar* d_inSynsyn1, __global scalar* d_inSynsyn2, __global scalar* d_inSynsyn3, __global scalar* d_inSynsyn4, __global scalar* d_inSynsyn5, __global scalar* d_inSynsyn6, __global scalar* d_inSynsyn7, __global scalar* d_inSynsyn8, __global scalar* d_inSynsyn9, __global scalar* d_shiftpost, __global scalar* d_shiftpre, __global scalar* d_xpost, __global scalar* d_xpre, volatile unsigned int spkQuePtrpre, float t) {
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
        
        if(id < 10) {
            scalar lx = d_xpost[id];
            scalar lshift = d_shiftpost[id];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn9 = d_inSynsyn9[id];
            Isyn += linSynsyn9; linSynsyn9 = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn8 = d_inSynsyn8[id];
            Isyn += linSynsyn8; linSynsyn8 = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn7 = d_inSynsyn7[id];
            Isyn += linSynsyn7; linSynsyn7 = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn6 = d_inSynsyn6[id];
            Isyn += linSynsyn6; linSynsyn6 = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn5 = d_inSynsyn5[id];
            Isyn += linSynsyn5; linSynsyn5 = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn4 = d_inSynsyn4[id];
            Isyn += linSynsyn4; linSynsyn4 = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn3 = d_inSynsyn3[id];
            Isyn += linSynsyn3; linSynsyn3 = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn2 = d_inSynsyn2[id];
            Isyn += linSynsyn2; linSynsyn2 = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn1 = d_inSynsyn1[id];
            Isyn += linSynsyn1; linSynsyn1 = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn0 = d_inSynsyn0[id];
            Isyn += linSynsyn0; linSynsyn0 = 0;
            // test whether spike condition was fulfilled previously
            const bool oldSpike= ((fmod(lx,(2.00000000000000000e+00f)) < 1e-4f));
            // calculate membrane potential
            lx= t+lshift;
            
            // test for and register a true spike
            if (((fmod(lx,(2.00000000000000000e+00f)) < 1e-4f)) && !(oldSpike)) {
                const unsigned int spkIdx = atomic_add(&shSpkCount, 1);
                shSpk[spkIdx] = id;
            }
            d_xpost[id] = lx;
            d_shiftpost[id] = lshift;
            // the post-synaptic dynamics
            
            d_inSynsyn9[id] = linSynsyn9;
            // the post-synaptic dynamics
            
            d_inSynsyn8[id] = linSynsyn8;
            // the post-synaptic dynamics
            
            d_inSynsyn7[id] = linSynsyn7;
            // the post-synaptic dynamics
            
            d_inSynsyn6[id] = linSynsyn6;
            // the post-synaptic dynamics
            
            d_inSynsyn5[id] = linSynsyn5;
            // the post-synaptic dynamics
            
            d_inSynsyn4[id] = linSynsyn4;
            // the post-synaptic dynamics
            
            d_inSynsyn3[id] = linSynsyn3;
            // the post-synaptic dynamics
            
            d_inSynsyn2[id] = linSynsyn2;
            // the post-synaptic dynamics
            
            d_inSynsyn1[id] = linSynsyn1;
            // the post-synaptic dynamics
            
            d_inSynsyn0[id] = linSynsyn0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomic_add(&d_glbSpkCntpost[0], shSpkCount);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < shSpkCount) {
            const unsigned int n = shSpk[localId];
            d_glbSpkpost[shPosSpk + localId] = n;
        }
    }
    
    // pre
    if(id >= 32 && id < 64) {
        const unsigned int lid = id - 32;
        const unsigned int readDelayOffset = (((spkQuePtrpre + 9) % 10) * 10);
        const unsigned int writeDelayOffset = (spkQuePtrpre * 10);
        
        if(lid < 10) {
            scalar lx = d_xpre[lid];
            scalar lshift = d_shiftpre[lid];
            
            // test whether spike condition was fulfilled previously
            const bool oldSpike= ((fmod(lx,(1.00000000000000000e+00f)) < 1e-4f));
            // calculate membrane potential
            lx= t+lshift;
            
            // test for and register a true spike
            if (((fmod(lx,(1.00000000000000000e+00f)) < 1e-4f)) && !(oldSpike)) {
                const unsigned int spkIdx = atomic_add(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
            }
            d_xpre[lid] = lx;
            d_shiftpre[lid] = lshift;
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
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(2, spkQuePtrpre));
    
    updateNeuronsKernel = cl::Kernel(updateNeuronsProgram, "updateNeuronsKernel");
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(0, DT));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(1, d_glbSpkCntpost));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(2, d_glbSpkCntpre));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(3, d_glbSpkpost));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(4, d_glbSpkpre));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(5, d_inSynsyn0));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(6, d_inSynsyn1));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(7, d_inSynsyn2));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(8, d_inSynsyn3));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(9, d_inSynsyn4));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(10, d_inSynsyn5));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(11, d_inSynsyn6));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(12, d_inSynsyn7));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(13, d_inSynsyn8));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(14, d_inSynsyn9));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(15, d_shiftpost));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(16, d_shiftpre));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(17, d_xpost));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(18, d_xpre));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(19, spkQuePtrpre));
}

void updateNeurons(float t) {
     {
        CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(2, spkQuePtrpre));
        const cl::NDRange globalWorkSize(32, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
     {
        CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(19, spkQuePtrpre));
        CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(20, t));
        
        const cl::NDRange globalWorkSize(64, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
