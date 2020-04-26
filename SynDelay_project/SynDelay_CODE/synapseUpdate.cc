#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateSynapsesProgramSrc = R"(typedef float scalar;

__kernel void updatePresynapticKernel(__global unsigned int* d_glbSpkCntInput, __global unsigned int* d_glbSpkCntInter, __global unsigned int* d_glbSpkInput, __global unsigned int* d_glbSpkInter, __global unsigned int* d_inSynInputInter, __global unsigned int* d_inSynInputOutput, __global unsigned int* d_inSynInterOutput, volatile unsigned int d_spkQuePtrInput, float t) {
    size_t groupId = get_group_id(0);
    size_t localId = get_group_id(0);
    const unsigned int id = 32 * groupId + localId; 
    __local unsigned int shSpk[32];
    // InputInter
    if(id < 512) {
        const unsigned int preReadDelaySlot = ((d_spkQuePtrInput + 4) % 7);
        const unsigned int preReadDelayOffset = preReadDelaySlot * 500;
        // only do this for existing neurons
        float linSyn = 0;
         {
            size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntInput[preReadDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkInput[preReadDelayOffset + (r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (id < 500) {
                        unsigned int synAddress = (shSpk[j] * 500) + id;
                        linSyn += (5.99999999999999978e-02f);
                    }
                }
            }
        }
        
        // only do this for existing neurons
        if (id < 500) {
            d_inSynInputInter[id] += linSyn;
        }
    }
    
    // InputOutput
    if(id >= 512 && id < 1024) {
        const unsigned int lid = id - 512;
        const unsigned int preReadDelaySlot = ((d_spkQuePtrInput + 1) % 7);
        const unsigned int preReadDelayOffset = preReadDelaySlot * 500;
        // only do this for existing neurons
        float linSyn = 0;
         {
            size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntInput[preReadDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkInput[preReadDelayOffset + (r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 500) {
                        unsigned int synAddress = (shSpk[j] * 500) + lid;
                        linSyn += (2.99999999999999989e-02f);
                    }
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 500) {
            d_inSynInputOutput[lid] += linSyn;
        }
    }
    
    // InterOutput
    if(id >= 1024 && id < 1536) {
        const unsigned int lid = id - 1024;
        // only do this for existing neurons
        float linSyn = 0;
         {
            size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntInter[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkInter[(r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 500) {
                        unsigned int synAddress = (shSpk[j] * 500) + lid;
                        linSyn += (2.99999999999999989e-02f);
                    }
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 500) {
            d_inSynInterOutput[lid] += linSyn;
        }
    }
    
}

)";

// Initialize the synapse update kernel(s)
void updateSynapsesProgramKernels() {
    updatePresynapticKernel = cl::Kernel(updateSynapsesProgram, "updatePresynapticKernel");
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(0, d_glbSpkCntInput));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(1, d_glbSpkCntInter));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(2, d_glbSpkInput));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(3, d_glbSpkInter));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(4, d_inSynInputInter));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(5, d_inSynInputOutput));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(6, d_inSynInterOutput));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(7, d_spkQuePtrInput));
}

void updateSynapses(float t) {
     {
        CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(8, t));
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updatePresynapticKernel, cl::NullRange, cl::NDRange(32)));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
