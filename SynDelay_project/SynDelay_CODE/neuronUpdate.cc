#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateNeuronsProgramSrc = R"(typedef float scalar;

__kernel void preNeuronResetKernel(__global unsigned int* d_glbSpkCntInput, __global unsigned int* d_glbSpkCntInter, __global unsigned int* d_glbSpkCntOutput, volatile unsigned int d_spkQuePtrInput) {
    size_t groupId = get_group_id(0);
    size_t localId = get_local_id(0);
    unsigned int id = 32 * groupId + localId;
    if(id == 0) {
        d_spkQuePtrInput = (d_spkQuePtrInput + 1) % 7;
        d_glbSpkCntInput[d_spkQuePtrInput] = 0;
    }
    else if(id == 1) {
        d_glbSpkCntInter[0] = 0;
    }
    else if(id == 2) {
        d_glbSpkCntOutput[0] = 0;
    }
}

__kernel void updateNeuronsKernel(const float DT, __global scalar* d_UInput, __global scalar* d_UInter, __global scalar* d_UOutput, __global scalar* d_VInput, __global scalar* d_VInter, __global scalar* d_VOutput, __global unsigned int* d_glbSpkCntInput, __global unsigned int* d_glbSpkCntInter, __global unsigned int* d_glbSpkCntOutput, __global unsigned int* d_glbSpkInput, __global unsigned int* d_glbSpkInter, __global unsigned int* d_glbSpkOutput, __global scalar* d_inSynInputInter, __global scalar* d_inSynInputOutput, __global scalar* d_inSynInterOutput, volatile unsigned int d_spkQuePtrInput, float t) {
    size_t groupId = get_group_id(0);
    size_t localId = get_local_id(0);
    const unsigned int id = 32 * groupId + localId; 
    volatile __local unsigned int shSpk[32];
    volatile __local unsigned int shPosSpk;
    volatile __local unsigned int shSpkCount;
    if (localId == 0); {
        shSpkCount = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    // Input
    if(id < 512) {
        const unsigned int readDelayOffset = (((d_spkQuePtrInput + 6) % 7) * 500);
        const unsigned int writeDelayOffset = (d_spkQuePtrInput * 500);
        
        if(id < 500) {
            scalar lV = d_VInput[id];
            scalar lU = d_UInput[id];
            
            float Isyn = 0;
            // current source InputCurrentSource
             {
                Isyn += (4.00000000000000000e+00f);
                
            }
            // test whether spike condition was fulfilled previously
            const bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
            if (lV >= 30.0f){
               lV=(-6.50000000000000000e+01f);
               lU+=(6.00000000000000000e+00f);
            } 
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT; //at two times for numerical stability
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT;
            lU+=(2.00000000000000004e-02f)*((2.00000000000000011e-01f)*lV-lU)*DT;
            if (lV > 30.0f){   //keep this to not confuse users with unrealistiv voltage values 
              lV=30.0f; 
            }
            
            // test for and register a true spike
            if ((lV >= 29.99f) && !(oldSpike)) {
                const unsigned int spkIdx = atomic_add(&shSpkCount, 1);
                shSpk[spkIdx] = id;
            }
            d_VInput[id] = lV;
            d_UInput[id] = lU;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomic_add(&d_glbSpkCntInput[d_spkQuePtrInput], shSpkCount);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < shSpkCount) {
            const unsigned int n = shSpk[localId];
            d_glbSpkInput[writeDelayOffset + shPosSpk + localId] = n;
        }
    }
    
    // Inter
    if(id >= 512 && id < 1024) {
        const unsigned int lid = id - 512;
        
        if(lid < 500) {
            scalar lV = d_VInter[lid];
            scalar lU = d_UInter[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynInputInter = d_inSynInputInter[lid];
            Isyn += linSynInputInter; linSynInputInter = 0;
            // test whether spike condition was fulfilled previously
            const bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
            if (lV >= 30.0f){
               lV=(-6.50000000000000000e+01f);
               lU+=(6.00000000000000000e+00f);
            } 
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT; //at two times for numerical stability
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT;
            lU+=(2.00000000000000004e-02f)*((2.00000000000000011e-01f)*lV-lU)*DT;
            if (lV > 30.0f){   //keep this to not confuse users with unrealistiv voltage values 
              lV=30.0f; 
            }
            
            // test for and register a true spike
            if ((lV >= 29.99f) && !(oldSpike)) {
                const unsigned int spkIdx = atomic_add(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
            }
            d_VInter[lid] = lV;
            d_UInter[lid] = lU;
            // the post-synaptic dynamics
            
            d_inSynInputInter[lid] = linSynInputInter;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomic_add(&d_glbSpkCntInter[0], shSpkCount);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < shSpkCount) {
            const unsigned int n = shSpk[localId];
            d_glbSpkInter[shPosSpk + localId] = n;
        }
    }
    
    // Output
    if(id >= 1024 && id < 1536) {
        const unsigned int lid = id - 1024;
        
        if(lid < 500) {
            scalar lV = d_VOutput[lid];
            scalar lU = d_UOutput[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynInterOutput = d_inSynInterOutput[lid];
            Isyn += linSynInterOutput; linSynInterOutput = 0;
            // pull inSyn values in a coalesced access
            float linSynInputOutput = d_inSynInputOutput[lid];
            Isyn += linSynInputOutput; linSynInputOutput = 0;
            // test whether spike condition was fulfilled previously
            const bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
            if (lV >= 30.0f){
               lV=(-6.50000000000000000e+01f);
               lU+=(6.00000000000000000e+00f);
            } 
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT; //at two times for numerical stability
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT;
            lU+=(2.00000000000000004e-02f)*((2.00000000000000011e-01f)*lV-lU)*DT;
            if (lV > 30.0f){   //keep this to not confuse users with unrealistiv voltage values 
              lV=30.0f; 
            }
            
            // test for and register a true spike
            if ((lV >= 29.99f) && !(oldSpike)) {
                const unsigned int spkIdx = atomic_add(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
            }
            d_VOutput[lid] = lV;
            d_UOutput[lid] = lU;
            // the post-synaptic dynamics
            
            d_inSynInterOutput[lid] = linSynInterOutput;
            // the post-synaptic dynamics
            
            d_inSynInputOutput[lid] = linSynInputOutput;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomic_add(&d_glbSpkCntOutput[0], shSpkCount);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < shSpkCount) {
            const unsigned int n = shSpk[localId];
            d_glbSpkOutput[shPosSpk + localId] = n;
        }
    }
    
}
)";

// Initialize the neuronUpdate kernels
void updateNeuronsProgramKernels() {
    preNeuronResetKernel = cl::Kernel(updateNeuronsProgram, "preNeuronResetKernel");
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(0, d_glbSpkCntInput));
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(1, d_glbSpkCntInter));
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(2, d_glbSpkCntOutput));
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(3, d_spkQuePtrInput));
    
    updateNeuronsKernel = cl::Kernel(updateNeuronsProgram, "updateNeuronsKernel");
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(0, DT));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(1, d_UInput));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(2, d_UInter));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(3, d_UOutput));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(4, d_VInput));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(5, d_VInter));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(6, d_VOutput));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(7, d_glbSpkCntInput));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(8, d_glbSpkCntInter));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(9, d_glbSpkCntOutput));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(10, d_glbSpkInput));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(11, d_glbSpkInter));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(12, d_glbSpkOutput));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(13, d_inSynInputInter));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(14, d_inSynInputOutput));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(15, d_inSynInterOutput));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(16, d_spkQuePtrInput));
}

void updateNeurons(float t) {
     {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, cl::NDRange(32)));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
     {
        CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(17, t));
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, cl::NDRange(32)));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
