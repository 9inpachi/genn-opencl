#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateSynapsesProgramSrc = R"(typedef float scalar;

__kernel void updatePostsynapticKernel(__global unsigned int* d_colLengthsyn, __global unsigned int* d_glbSpkCntpost, __global unsigned int* d_glbSpkpost, __global unsigned int* d_remapsyn, __global scalar* d_ssyn, __global scalar* d_wsyn, volatile unsigned int spkQuePtrpost, float t) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    __local unsigned int shSpk[32];
    __local unsigned int shColLength[32];
    // syn
    if(id < 32) {
        const unsigned int postReadDelaySlot = ((spkQuePtrpost + 1) % 21);
        const unsigned int postReadDelayOffset = postReadDelaySlot * 10;
        const unsigned int numSpikes = d_glbSpkCntpost[postReadDelaySlot];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (localId < numSpikesInBlock) {
                const unsigned int spk = d_glbSpkpost[postReadDelayOffset + (r * 32) + localId];
                shSpk[localId] = spk;
                shColLength[localId] = d_colLengthsyn[spk];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // only work on existing neurons
            if (id < 1) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    if (id < shColLength[j]) {
                        const unsigned int synAddress = d_remapsyn[(shSpk[j] * 1) + id];
                        const unsigned int ipre = synAddress / 1;
                        d_wsyn[synAddress]= d_ssyn[postReadDelayOffset + shSpk[j]];
                    }
                }
            }
        }
    }
    
}

)";

// Initialize the synapse update kernel(s)
void updateSynapsesProgramKernels() {
    updatePostsynapticKernel = cl::Kernel(updateSynapsesProgram, "updatePostsynapticKernel");
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(0, d_colLengthsyn));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(1, d_glbSpkCntpost));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(2, d_glbSpkpost));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(3, d_remapsyn));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(4, d_ssyn));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(5, d_wsyn));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(6, spkQuePtrpost));
}

void updateSynapses(float t) {
     {
        CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(6, spkQuePtrpost));
        CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(7, t));
        
        const cl::NDRange globalWorkSize(32, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updatePostsynapticKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
