#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateSynapsesProgramSrc = R"(typedef float scalar;

void atomic_add_f_global(volatile __global float *source, const float operand) {
    union { unsigned int intVal; float floatVal; } newVal;
    union { unsigned int intVal; float floatVal; } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    }
    while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void updatePresynapticKernel(__global unsigned int* d_glbSpkCntpre, __global unsigned int* d_glbSpkpre, __global float* d_inSynsyn0, __global float* d_inSynsyn1, __global float* d_inSynsyn2, __global float* d_inSynsyn3, __global float* d_inSynsyn4, __global float* d_inSynsyn5, __global float* d_inSynsyn6, __global float* d_inSynsyn7, __global float* d_inSynsyn8, __global float* d_inSynsyn9, __global scalar* d_wsyn0, __global scalar* d_wsyn1, __global scalar* d_wsyn2, __global scalar* d_wsyn3, __global scalar* d_wsyn4, __global scalar* d_wsyn5, __global scalar* d_wsyn6, __global scalar* d_wsyn7, __global scalar* d_wsyn8, __global scalar* d_wsyn9, __global scalar* d_xpre, volatile unsigned int spkQuePtrpre, float t) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    __local unsigned int shSpk[32];
    // syn0
    if(id < 32) {
        const unsigned int preReadDelaySlot = spkQuePtrpre;
        const unsigned int preReadDelayOffset = preReadDelaySlot * 10;
        // only do this for existing neurons
        float linSyn = 0;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntpre[preReadDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkpre[preReadDelayOffset + (r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (id < 10) {
                        unsigned int synAddress = (shSpk[j] * 10) + id;
                        d_wsyn0[synAddress]= d_xpre[preReadDelayOffset + shSpk[j]];}
                }
            }
        }
        
        // only do this for existing neurons
        if (id < 10) {
            d_inSynsyn0[id] += linSyn;
        }
    }
    
    // syn1
    if(id >= 32 && id < 64) {
        const unsigned int lid = id - 32;
        const unsigned int preReadDelaySlot = ((spkQuePtrpre + 9) % 10);
        const unsigned int preReadDelayOffset = preReadDelaySlot * 10;
        // only do this for existing neurons
        float linSyn = 0;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntpre[preReadDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkpre[preReadDelayOffset + (r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 10) {
                        unsigned int synAddress = (shSpk[j] * 10) + lid;
                        d_wsyn1[synAddress]= d_xpre[preReadDelayOffset + shSpk[j]];}
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 10) {
            d_inSynsyn1[lid] += linSyn;
        }
    }
    
    // syn2
    if(id >= 64 && id < 96) {
        const unsigned int lid = id - 64;
        const unsigned int preReadDelaySlot = ((spkQuePtrpre + 8) % 10);
        const unsigned int preReadDelayOffset = preReadDelaySlot * 10;
        // only do this for existing neurons
        float linSyn = 0;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntpre[preReadDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkpre[preReadDelayOffset + (r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 10) {
                        unsigned int synAddress = (shSpk[j] * 10) + lid;
                        d_wsyn2[synAddress]= d_xpre[preReadDelayOffset + shSpk[j]];}
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 10) {
            d_inSynsyn2[lid] += linSyn;
        }
    }
    
    // syn3
    if(id >= 96 && id < 128) {
        const unsigned int lid = id - 96;
        const unsigned int preReadDelaySlot = ((spkQuePtrpre + 7) % 10);
        const unsigned int preReadDelayOffset = preReadDelaySlot * 10;
        // only do this for existing neurons
        float linSyn = 0;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntpre[preReadDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkpre[preReadDelayOffset + (r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 10) {
                        unsigned int synAddress = (shSpk[j] * 10) + lid;
                        d_wsyn3[synAddress]= d_xpre[preReadDelayOffset + shSpk[j]];}
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 10) {
            d_inSynsyn3[lid] += linSyn;
        }
    }
    
    // syn4
    if(id >= 128 && id < 160) {
        const unsigned int lid = id - 128;
        const unsigned int preReadDelaySlot = ((spkQuePtrpre + 6) % 10);
        const unsigned int preReadDelayOffset = preReadDelaySlot * 10;
        // only do this for existing neurons
        float linSyn = 0;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntpre[preReadDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkpre[preReadDelayOffset + (r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 10) {
                        unsigned int synAddress = (shSpk[j] * 10) + lid;
                        d_wsyn4[synAddress]= d_xpre[preReadDelayOffset + shSpk[j]];}
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 10) {
            d_inSynsyn4[lid] += linSyn;
        }
    }
    
    // syn5
    if(id >= 160 && id < 192) {
        const unsigned int lid = id - 160;
        const unsigned int preReadDelaySlot = ((spkQuePtrpre + 5) % 10);
        const unsigned int preReadDelayOffset = preReadDelaySlot * 10;
        // only do this for existing neurons
        float linSyn = 0;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntpre[preReadDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkpre[preReadDelayOffset + (r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 10) {
                        unsigned int synAddress = (shSpk[j] * 10) + lid;
                        d_wsyn5[synAddress]= d_xpre[preReadDelayOffset + shSpk[j]];}
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 10) {
            d_inSynsyn5[lid] += linSyn;
        }
    }
    
    // syn6
    if(id >= 192 && id < 224) {
        const unsigned int lid = id - 192;
        const unsigned int preReadDelaySlot = ((spkQuePtrpre + 4) % 10);
        const unsigned int preReadDelayOffset = preReadDelaySlot * 10;
        // only do this for existing neurons
        float linSyn = 0;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntpre[preReadDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkpre[preReadDelayOffset + (r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 10) {
                        unsigned int synAddress = (shSpk[j] * 10) + lid;
                        d_wsyn6[synAddress]= d_xpre[preReadDelayOffset + shSpk[j]];}
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 10) {
            d_inSynsyn6[lid] += linSyn;
        }
    }
    
    // syn7
    if(id >= 224 && id < 256) {
        const unsigned int lid = id - 224;
        const unsigned int preReadDelaySlot = ((spkQuePtrpre + 3) % 10);
        const unsigned int preReadDelayOffset = preReadDelaySlot * 10;
        // only do this for existing neurons
        float linSyn = 0;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntpre[preReadDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkpre[preReadDelayOffset + (r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 10) {
                        unsigned int synAddress = (shSpk[j] * 10) + lid;
                        d_wsyn7[synAddress]= d_xpre[preReadDelayOffset + shSpk[j]];}
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 10) {
            d_inSynsyn7[lid] += linSyn;
        }
    }
    
    // syn8
    if(id >= 256 && id < 288) {
        const unsigned int lid = id - 256;
        const unsigned int preReadDelaySlot = ((spkQuePtrpre + 2) % 10);
        const unsigned int preReadDelayOffset = preReadDelaySlot * 10;
        // only do this for existing neurons
        float linSyn = 0;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntpre[preReadDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkpre[preReadDelayOffset + (r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 10) {
                        unsigned int synAddress = (shSpk[j] * 10) + lid;
                        d_wsyn8[synAddress]= d_xpre[preReadDelayOffset + shSpk[j]];}
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 10) {
            d_inSynsyn8[lid] += linSyn;
        }
    }
    
    // syn9
    if(id >= 288 && id < 320) {
        const unsigned int lid = id - 288;
        const unsigned int preReadDelaySlot = ((spkQuePtrpre + 1) % 10);
        const unsigned int preReadDelayOffset = preReadDelaySlot * 10;
        // only do this for existing neurons
        float linSyn = 0;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntpre[preReadDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkpre[preReadDelayOffset + (r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 10) {
                        unsigned int synAddress = (shSpk[j] * 10) + lid;
                        d_wsyn9[synAddress]= d_xpre[preReadDelayOffset + shSpk[j]];}
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 10) {
            d_inSynsyn9[lid] += linSyn;
        }
    }
    
}

)";

// Initialize the synapse update kernel(s)
void updateSynapsesProgramKernels() {
    updatePresynapticKernel = cl::Kernel(updateSynapsesProgram, "updatePresynapticKernel");
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(0, d_glbSpkCntpre));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(1, d_glbSpkpre));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(2, d_inSynsyn0));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(3, d_inSynsyn1));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(4, d_inSynsyn2));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(5, d_inSynsyn3));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(6, d_inSynsyn4));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(7, d_inSynsyn5));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(8, d_inSynsyn6));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(9, d_inSynsyn7));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(10, d_inSynsyn8));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(11, d_inSynsyn9));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(12, d_wsyn0));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(13, d_wsyn1));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(14, d_wsyn2));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(15, d_wsyn3));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(16, d_wsyn4));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(17, d_wsyn5));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(18, d_wsyn6));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(19, d_wsyn7));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(20, d_wsyn8));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(21, d_wsyn9));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(22, d_xpre));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(23, spkQuePtrpre));
}

void updateSynapses(float t) {
     {
        CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(23, spkQuePtrpre));
        CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(24, t));
        
        const cl::NDRange globalWorkSize(320, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updatePresynapticKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
