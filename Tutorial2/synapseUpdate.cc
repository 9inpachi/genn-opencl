#include "definitionsInternal.h"

// Update synapses kernel
extern "C" const char* updateSynapsesKernelSource = R"(typedef float scalar;

void atomic_add(
    volatile __global float *source,
    const float operand
) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
 
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void updatePresynapticKernel(
    __global unsigned int* dd_glbSpkCntExc,
    __global unsigned int* dd_glbSpkExc,
    __global unsigned int* dd_rowLengthExc_Exc,
    __global unsigned int* dd_indExc_Exc,
    __global float* dd_inSynExc_Exc,
    __global unsigned int* dd_rowLengthExc_Inh,
    __global unsigned int* dd_indExc_Inh,
    __global float* dd_inSynExc_Inh,
    __global unsigned int* dd_glbSpkCntInh,
    __global unsigned int* dd_glbSpkInh,
    __global unsigned int* dd_rowLengthInh_Exc,
    __global unsigned int* dd_indInh_Exc,
    __global float* dd_inSynInh_Exc,
    __global unsigned int* dd_rowLengthInh_Inh,
    __global unsigned int* dd_indInh_Inh,
    __global float* dd_inSynInh_Inh,
    float t
) {
    size_t groupId = get_group_id(0);
    size_t localId = get_local_id(0);
    const unsigned int id = 32 * groupId + localId; 
    __local unsigned int shRowLength[32];
    __local unsigned int shSpk[32];
    // Exc_Exc
    if(id < 960) {
         {
            const unsigned int numSpikes = dd_glbSpkCntExc[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localId < numSpikesInBlock) {
                    const unsigned int spk = dd_glbSpkExc[(r * 32) + localId];
                    shSpk[localId] = spk;
                    shRowLength[localId] = dd_rowLengthExc_Exc[spk];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (id < 953) {
                        unsigned int synAddress = shSpk[j] * 953;
                        const unsigned int npost = shRowLength[j];
                        if (id < npost) {
                            synAddress += id;
                            const unsigned int ipost = dd_indExc_Exc[synAddress];
                            atomic_add(&dd_inSynExc_Exc[ipost], (5.00000000000000028e-02f));
                        }
                    }
                }
            }
        }
        
    }
    
    // Exc_Inh
    if(id >= 960 && id < 1248) {
        const unsigned int lid = id - 960;
         {
            const unsigned int numSpikes = dd_glbSpkCntExc[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localId < numSpikesInBlock) {
                    const unsigned int spk = dd_glbSpkExc[(r * 32) + localId];
                    shSpk[localId] = spk;
                    shRowLength[localId] = dd_rowLengthExc_Inh[spk];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 279) {
                        unsigned int synAddress = shSpk[j] * 279;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            synAddress += lid;
                            const unsigned int ipost = dd_indExc_Inh[synAddress];
                            atomic_add(&dd_inSynExc_Inh[ipost], (5.00000000000000028e-02f));
                        }
                    }
                }
            }
        }
        
    }
    
    // Inh_Exc
    if(id >= 1248 && id < 2208) {
        const unsigned int lid = id - 1248;
         {
            const unsigned int numSpikes = dd_glbSpkCntInh[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localId < numSpikesInBlock) {
                    const unsigned int spk = dd_glbSpkInh[(r * 32) + localId];
                    shSpk[localId] = spk;
                    shRowLength[localId] = dd_rowLengthInh_Exc[spk];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 946) {
                        unsigned int synAddress = shSpk[j] * 946;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            synAddress += lid;
                            const unsigned int ipost = dd_indInh_Exc[synAddress];
                            atomic_add(&dd_inSynInh_Exc[ipost], (-2.50000000000000000e-01f));
                        }
                    }
                }
            }
        }
        
    }
    
    // Inh_Inh
    if(id >= 2208 && id < 2496) {
        const unsigned int lid = id - 2208;
         {
            const unsigned int numSpikes = dd_glbSpkCntInh[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localId < numSpikesInBlock) {
                    const unsigned int spk = dd_glbSpkInh[(r * 32) + localId];
                    shSpk[localId] = spk;
                    shRowLength[localId] = dd_rowLengthInh_Inh[spk];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 275) {
                        unsigned int synAddress = shSpk[j] * 275;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            synAddress += lid;
                            const unsigned int ipost = dd_indInh_Inh[synAddress];
                            atomic_add(&dd_inSynInh_Inh[ipost], (-2.50000000000000000e-01f));
                        }
                    }
                }
            }
        }
        
    }
}

)";

// Initialize the synapses update kernels
void initUpdateSynapsesKernels() {
    updatePresynapticKernel = cl::Kernel(updateSynapsesProgram, "updatePresynapticKernel");
    cl_int err;
    err = updatePresynapticKernel.setArg(0, d_glbSpkCntExc);
    updatePresynapticKernel.setArg(1, d_glbSpkExc);
    updatePresynapticKernel.setArg(2, d_rowLengthExc_Exc);
    updatePresynapticKernel.setArg(3, d_indExc_Exc);
    updatePresynapticKernel.setArg(4, d_inSynExc_Exc);
    updatePresynapticKernel.setArg(5, d_rowLengthExc_Inh);
    updatePresynapticKernel.setArg(6, d_indExc_Inh);
    updatePresynapticKernel.setArg(7, d_inSynExc_Inh);
    updatePresynapticKernel.setArg(8, d_glbSpkCntInh);
    updatePresynapticKernel.setArg(9, d_glbSpkInh);
    updatePresynapticKernel.setArg(10, d_rowLengthInh_Exc);
    updatePresynapticKernel.setArg(11, d_indInh_Exc);
    updatePresynapticKernel.setArg(12, d_inSynInh_Exc);
    updatePresynapticKernel.setArg(13, d_rowLengthInh_Inh);
    updatePresynapticKernel.setArg(14, d_indInh_Inh);
    updatePresynapticKernel.setArg(15, d_inSynInh_Inh);
}

void updateSynapses(float t) {
    updatePresynapticKernel.setArg(16, t);
    commandQueue.enqueueNDRangeKernel(updatePresynapticKernel, cl::NullRange, cl::NDRange(78), cl::NDRange(32));
    commandQueue.finish();
}
