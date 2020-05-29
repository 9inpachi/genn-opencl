#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateSynapsesProgramSrc = R"(typedef float scalar;

// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

void atomic_add_f_global(volatile __global float *source, const float operand) {
    union { unsigned int intVal; float floatVal; } newVal;
    union { unsigned int intVal; float floatVal; } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    }
    while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void preSynapseResetKernel(volatile unsigned int denDelayPtrSyn) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    if(id == 0) {
        denDelayPtrSyn = (denDelayPtrSyn + 1) % 10;
    }
}

__kernel void updatePresynapticKernel(__global unsigned char* d_dSyn, __global scalar* d_denDelaySyn, __global scalar* d_gSyn, __global unsigned int* d_glbSpkCntPre, __global unsigned int* d_glbSpkPre, __global float* d_inSynSyn, volatile unsigned int denDelayPtrSyn, float t) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    __local unsigned int shSpk[32];
    // Syn
    if(id < 32) {
        // only do this for existing neurons
        float linSyn = 0;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntPre[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkPre[(r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (id < 1) {
                        unsigned int synAddress = (shSpk[j] * 1) + id;
                        atomic_add_f_global(&d_denDelaySyn[(((denDelayPtrSyn + d_dSyn[synAddress]) % 10) * 1) + id], d_gSyn[synAddress]);
                    }
                }
            }
        }
        
        // only do this for existing neurons
        if (id < 1) {
            d_inSynSyn[id] += linSyn;
        }
    }
    
}

)";

// Initialize the synapse update kernel(s)
void updateSynapsesProgramKernels() {
    preSynapseResetKernel = cl::Kernel(updateSynapsesProgram, "preSynapseResetKernel");
    CHECK_OPENCL_ERRORS(preSynapseResetKernel.setArg(0, denDelayPtrSyn));
    updatePresynapticKernel = cl::Kernel(updateSynapsesProgram, "updatePresynapticKernel");
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(0, d_dSyn));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(1, d_denDelaySyn));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(2, d_gSyn));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(3, d_glbSpkCntPre));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(4, d_glbSpkPre));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(5, d_inSynSyn));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(6, denDelayPtrSyn));
}

void updateSynapses(float t) {
     {
        CHECK_OPENCL_ERRORS(preSynapseResetKernel.setArg(0, denDelayPtrSyn));
        const cl::NDRange globalWorkSize(32, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(preSynapseResetKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
     {
        CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(6, denDelayPtrSyn));
        CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(7, t));
        
        const cl::NDRange globalWorkSize(32, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updatePresynapticKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
