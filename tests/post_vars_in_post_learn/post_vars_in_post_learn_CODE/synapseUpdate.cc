#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateSynapsesProgramSrc = R"(typedef float scalar;

__kernel void updatePostsynapticKernel(__global unsigned int* d_glbSpkCntpost, __global unsigned int* d_glbSpkpost, __global scalar* d_spkQuePtrpre, __global scalar* d_wsyn0, __global scalar* d_wsyn1, __global scalar* d_wsyn2, __global scalar* d_wsyn3, __global scalar* d_wsyn4, __global scalar* d_wsyn5, __global scalar* d_wsyn6, __global scalar* d_wsyn7, __global scalar* d_wsyn8, __global scalar* d_wsyn9, __global scalar* d_xpost, volatile unsigned int spkQuePtrpre, float t) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    __local unsigned int shSpk[32];
    // syn0
    if(id < 32) {
        const unsigned int preReadDelayOffset = d_spkQuePtrpre * 10;
        const unsigned int numSpikes = d_glbSpkCntpost[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (localId < numSpikesInBlock) {
                const unsigned int spk = d_glbSpkpost[(r * 32) + localId];
                shSpk[localId] = spk;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // only work on existing neurons
            if (id < 10) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    const unsigned int synAddress = (id * 10) + shSpk[j];
                    d_wsyn0[synAddress]= d_xpost[shSpk[j]];}
            }
        }
    }
    
    // syn1
    if(id >= 32 && id < 64) {
        const unsigned int lid = id - 32;
        const unsigned int preReadDelayOffset = ((d_spkQuePtrpre + 9) % 10) * 10;
        const unsigned int numSpikes = d_glbSpkCntpost[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (localId < numSpikesInBlock) {
                const unsigned int spk = d_glbSpkpost[(r * 32) + localId];
                shSpk[localId] = spk;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // only work on existing neurons
            if (lid < 10) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    const unsigned int synAddress = (lid * 10) + shSpk[j];
                    d_wsyn1[synAddress]= d_xpost[shSpk[j]];}
            }
        }
    }
    
    // syn2
    if(id >= 64 && id < 96) {
        const unsigned int lid = id - 64;
        const unsigned int preReadDelayOffset = ((d_spkQuePtrpre + 8) % 10) * 10;
        const unsigned int numSpikes = d_glbSpkCntpost[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (localId < numSpikesInBlock) {
                const unsigned int spk = d_glbSpkpost[(r * 32) + localId];
                shSpk[localId] = spk;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // only work on existing neurons
            if (lid < 10) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    const unsigned int synAddress = (lid * 10) + shSpk[j];
                    d_wsyn2[synAddress]= d_xpost[shSpk[j]];}
            }
        }
    }
    
    // syn3
    if(id >= 96 && id < 128) {
        const unsigned int lid = id - 96;
        const unsigned int preReadDelayOffset = ((d_spkQuePtrpre + 7) % 10) * 10;
        const unsigned int numSpikes = d_glbSpkCntpost[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (localId < numSpikesInBlock) {
                const unsigned int spk = d_glbSpkpost[(r * 32) + localId];
                shSpk[localId] = spk;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // only work on existing neurons
            if (lid < 10) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    const unsigned int synAddress = (lid * 10) + shSpk[j];
                    d_wsyn3[synAddress]= d_xpost[shSpk[j]];}
            }
        }
    }
    
    // syn4
    if(id >= 128 && id < 160) {
        const unsigned int lid = id - 128;
        const unsigned int preReadDelayOffset = ((d_spkQuePtrpre + 6) % 10) * 10;
        const unsigned int numSpikes = d_glbSpkCntpost[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (localId < numSpikesInBlock) {
                const unsigned int spk = d_glbSpkpost[(r * 32) + localId];
                shSpk[localId] = spk;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // only work on existing neurons
            if (lid < 10) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    const unsigned int synAddress = (lid * 10) + shSpk[j];
                    d_wsyn4[synAddress]= d_xpost[shSpk[j]];}
            }
        }
    }
    
    // syn5
    if(id >= 160 && id < 192) {
        const unsigned int lid = id - 160;
        const unsigned int preReadDelayOffset = ((d_spkQuePtrpre + 5) % 10) * 10;
        const unsigned int numSpikes = d_glbSpkCntpost[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (localId < numSpikesInBlock) {
                const unsigned int spk = d_glbSpkpost[(r * 32) + localId];
                shSpk[localId] = spk;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // only work on existing neurons
            if (lid < 10) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    const unsigned int synAddress = (lid * 10) + shSpk[j];
                    d_wsyn5[synAddress]= d_xpost[shSpk[j]];}
            }
        }
    }
    
    // syn6
    if(id >= 192 && id < 224) {
        const unsigned int lid = id - 192;
        const unsigned int preReadDelayOffset = ((d_spkQuePtrpre + 4) % 10) * 10;
        const unsigned int numSpikes = d_glbSpkCntpost[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (localId < numSpikesInBlock) {
                const unsigned int spk = d_glbSpkpost[(r * 32) + localId];
                shSpk[localId] = spk;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // only work on existing neurons
            if (lid < 10) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    const unsigned int synAddress = (lid * 10) + shSpk[j];
                    d_wsyn6[synAddress]= d_xpost[shSpk[j]];}
            }
        }
    }
    
    // syn7
    if(id >= 224 && id < 256) {
        const unsigned int lid = id - 224;
        const unsigned int preReadDelayOffset = ((d_spkQuePtrpre + 3) % 10) * 10;
        const unsigned int numSpikes = d_glbSpkCntpost[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (localId < numSpikesInBlock) {
                const unsigned int spk = d_glbSpkpost[(r * 32) + localId];
                shSpk[localId] = spk;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // only work on existing neurons
            if (lid < 10) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    const unsigned int synAddress = (lid * 10) + shSpk[j];
                    d_wsyn7[synAddress]= d_xpost[shSpk[j]];}
            }
        }
    }
    
    // syn8
    if(id >= 256 && id < 288) {
        const unsigned int lid = id - 256;
        const unsigned int preReadDelayOffset = ((d_spkQuePtrpre + 2) % 10) * 10;
        const unsigned int numSpikes = d_glbSpkCntpost[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (localId < numSpikesInBlock) {
                const unsigned int spk = d_glbSpkpost[(r * 32) + localId];
                shSpk[localId] = spk;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // only work on existing neurons
            if (lid < 10) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    const unsigned int synAddress = (lid * 10) + shSpk[j];
                    d_wsyn8[synAddress]= d_xpost[shSpk[j]];}
            }
        }
    }
    
    // syn9
    if(id >= 288 && id < 320) {
        const unsigned int lid = id - 288;
        const unsigned int preReadDelayOffset = ((d_spkQuePtrpre + 1) % 10) * 10;
        const unsigned int numSpikes = d_glbSpkCntpost[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (localId < numSpikesInBlock) {
                const unsigned int spk = d_glbSpkpost[(r * 32) + localId];
                shSpk[localId] = spk;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // only work on existing neurons
            if (lid < 10) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    const unsigned int synAddress = (lid * 10) + shSpk[j];
                    d_wsyn9[synAddress]= d_xpost[shSpk[j]];}
            }
        }
    }
    
}

)";

// Initialize the synapse update kernel(s)
void updateSynapsesProgramKernels() {
    updatePostsynapticKernel = cl::Kernel(updateSynapsesProgram, "updatePostsynapticKernel");
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(0, d_glbSpkCntpost));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(1, d_glbSpkpost));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(2, d_spkQuePtrpre));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(3, d_wsyn0));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(4, d_wsyn1));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(5, d_wsyn2));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(6, d_wsyn3));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(7, d_wsyn4));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(8, d_wsyn5));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(9, d_wsyn6));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(10, d_wsyn7));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(11, d_wsyn8));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(12, d_wsyn9));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(13, d_xpost));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(14, spkQuePtrpre));
}

void updateSynapses(float t) {
     {
        CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(14, spkQuePtrpre));
        CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(15, t));
        
        const cl::NDRange globalWorkSize(320, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updatePostsynapticKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
