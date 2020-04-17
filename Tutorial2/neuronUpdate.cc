#include "definitionsInternal.h"

// Update neurons kernel
extern "C" const char* updateNeuronsKernelSource = R"(typedef float scalar;

__kernel void preNeuronResetKernel(
    __global unsigned int* dd_glbSpkCntExc,
    __global unsigned int* dd_glbSpkCntInh
) {
    size_t groupId = get_group_id(0);
    size_t localId = get_local_id(0);
    unsigned int id = 32 * groupId + localId;
    if(id == 0) {
        dd_glbSpkCntExc[0] = 0;
    }
    else if(id == 1) {
        dd_glbSpkCntInh[0] = 0;
    }
}

__kernel void updateNeuronsKernel(
    __global unsigned int* dd_glbSpkCntExc,
    __global unsigned int* dd_glbSpkExc,
    __global scalar* dd_VExc,
    __global scalar* dd_UExc,
    __global float* dd_inSynInh_Exc,
    __global float* dd_inSynExc_Exc,
    __global unsigned int* dd_glbSpkCntInh,
    __global unsigned int* dd_glbSpkInh,
    __global scalar* dd_VInh,
    __global scalar* dd_UInh,
    __global float* dd_inSynInh_Inh,
    __global float* dd_inSynExc_Inh,
    const float DT,
    float t
) {
    size_t groupId = get_group_id(0);
    size_t localId = get_local_id(0);
    const unsigned int id = 64 * groupId + localId; 
    volatile __local unsigned int shSpk[64];
    volatile __local unsigned int shPosSpk;
    volatile __local unsigned int shSpkCount;
    if (localId == 0); {
        shSpkCount = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    // Exc
    if(id < 8000) {
        
        if(id < 8000) {
            scalar lV = dd_VExc[id];
            scalar lU = dd_UExc[id];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynInh_Exc = dd_inSynInh_Exc[id];
            Isyn += linSynInh_Exc; linSynInh_Exc = 0;
            // pull inSyn values in a coalesced access
            float linSynExc_Exc = dd_inSynExc_Exc[id];
            Isyn += linSynExc_Exc; linSynExc_Exc = 0;
            // current source ExcStim
             {
                Isyn += (6.00000000000000000e+00f);
                
            }
            // test whether spike condition was fulfilled previously
            const bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
            if (lV >= 30.0f){
               lV=(-6.50000000000000000e+01f);
               lU+=(8.00000000000000000e+00f);
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
            dd_VExc[id] = lV;
            dd_UExc[id] = lU;
            // the post-synaptic dynamics
            
            dd_inSynInh_Exc[id] = linSynInh_Exc;
            // the post-synaptic dynamics
            
            dd_inSynExc_Exc[id] = linSynExc_Exc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomic_add(&dd_glbSpkCntExc[0], shSpkCount);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < shSpkCount) {
            const unsigned int n = shSpk[localId];
            dd_glbSpkExc[shPosSpk + localId] = n;
        }
    }
    
    // Inh
    if(id >= 8000 && id < 10048) {
        const unsigned int lid = id - 8000;
        
        if(lid < 2000) {
            scalar lV = dd_VInh[lid];
            scalar lU = dd_UInh[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynInh_Inh = dd_inSynInh_Inh[lid];
            Isyn += linSynInh_Inh; linSynInh_Inh = 0;
            // pull inSyn values in a coalesced access
            float linSynExc_Inh = dd_inSynExc_Inh[lid];
            Isyn += linSynExc_Inh; linSynExc_Inh = 0;
            // current source InhStim
             {
                Isyn += (6.00000000000000000e+00f);
                
            }
            // test whether spike condition was fulfilled previously
            const bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
            if (lV >= 30.0f){
               lV=(-6.50000000000000000e+01f);
               lU+=(8.00000000000000000e+00f);
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
            dd_VInh[lid] = lV;
            dd_UInh[lid] = lU;
            // the post-synaptic dynamics
            
            dd_inSynInh_Inh[lid] = linSynInh_Inh;
            // the post-synaptic dynamics
            
            dd_inSynExc_Inh[lid] = linSynExc_Inh;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomic_add(&dd_glbSpkCntInh[0], shSpkCount);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < shSpkCount) {
            const unsigned int n = shSpk[localId];
            dd_glbSpkInh[shPosSpk + localId] = n;
        }
    }
}

)";

// Initialize the neuronUpdate kernels
void initUpdateNeuronsKernels() {
    preNeuronResetKernel = cl::Kernel(updateNeuronsProgram, "preNeuronResetKernel");
    preNeuronResetKernel.setArg(0, d_glbSpkCntExc);
    preNeuronResetKernel.setArg(1, d_glbSpkCntInh);

    updateNeuronsKernel = cl::Kernel(updateNeuronsProgram, "updateNeuronsKernel");
    preNeuronResetKernel.setArg(0, d_glbSpkCntExc);
    preNeuronResetKernel.setArg(1, d_glbSpkExc);
    preNeuronResetKernel.setArg(2, d_VExc);
    preNeuronResetKernel.setArg(3, d_UExc);
    preNeuronResetKernel.setArg(4, d_inSynInh_Exc);
    preNeuronResetKernel.setArg(5, d_inSynExc_Exc);
    preNeuronResetKernel.setArg(6, d_glbSpkCntInh);
    preNeuronResetKernel.setArg(7, d_glbSpkInh);
    preNeuronResetKernel.setArg(8, d_VInh);
    preNeuronResetKernel.setArg(9, d_UInh);
    preNeuronResetKernel.setArg(10, d_inSynInh_Inh);
    preNeuronResetKernel.setArg(11, d_inSynExc_Inh);
    preNeuronResetKernel.setArg(12, DT);
}

void updateNeurons(float t) {
    commandQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, cl::NDRange(1, 1), cl::NDRange(32, 1));
    commandQueue.finish();

    updateNeuronsKernel.setArg(13, t);
    commandQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, cl::NDRange(157, 1), cl::NDRange(64, 1));
    commandQueue.finish();
}