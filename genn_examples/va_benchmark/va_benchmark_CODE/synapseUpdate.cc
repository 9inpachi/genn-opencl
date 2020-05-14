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

__kernel void updatePresynapticKernel(__global unsigned int* d_glbSpkCntE, __global unsigned int* d_glbSpkCntI, __global unsigned int* d_glbSpkE, __global unsigned int* d_glbSpkI, __global float* d_inSynEE, __global float* d_inSynEI, __global float* d_inSynIE, __global float* d_inSynII, __global unsigned int* d_indEE, __global unsigned int* d_indEI, __global unsigned int* d_indIE, __global unsigned int* d_indII, __global unsigned int* d_rowLengthEE, __global unsigned int* d_rowLengthEI, __global unsigned int* d_rowLengthIE, __global unsigned int* d_rowLengthII, float t) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    __local unsigned int shRowLength[32];
    __local unsigned int shSpk[32];
    // EE
    if(id < 4384) {
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntE[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkE[(r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                    shRowLength[localIdi] = d_rowLengthEE[spk];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (id < 4355) {
                        unsigned int synAddress = shSpk[j] * 4355;
                        const unsigned int npost = shRowLength[j];
                        if (id < npost) {
                            synAddress += id;
                            const unsigned int ipost = d_indEE[synAddress];
                            atomic_add_f_global(&d_inSynEE[ipost], (6.39999999999999971e-05f));
                        }
                    }
                }
            }
        }
        
    }
    
    // EI
    if(id >= 4384 && id < 5568) {
        const unsigned int lid = id - 4384;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntE[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkE[(r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                    shRowLength[localIdi] = d_rowLengthEI[spk];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 1180) {
                        unsigned int synAddress = shSpk[j] * 1180;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            synAddress += lid;
                            const unsigned int ipost = d_indEI[synAddress];
                            atomic_add_f_global(&d_inSynEI[ipost], (6.39999999999999971e-05f));
                        }
                    }
                }
            }
        }
        
    }
    
    // IE
    if(id >= 5568 && id < 9920) {
        const unsigned int lid = id - 5568;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntI[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkI[(r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                    shRowLength[localIdi] = d_rowLengthIE[spk];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 4341) {
                        unsigned int synAddress = shSpk[j] * 4341;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            synAddress += lid;
                            const unsigned int ipost = d_indIE[synAddress];
                            atomic_add_f_global(&d_inSynIE[ipost], (-8.15999999999999994e-04f));
                        }
                    }
                }
            }
        }
        
    }
    
    // II
    if(id >= 9920 && id < 11104) {
        const unsigned int lid = id - 9920;
         {
            const size_t localIdi = get_local_id(0);
            const unsigned int numSpikes = d_glbSpkCntI[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (localIdi < numSpikesInBlock) {
                    const unsigned int spk = d_glbSpkI[(r * 32) + localIdi];
                    shSpk[localIdi] = spk;
                    shRowLength[localIdi] = d_rowLengthII[spk];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 1172) {
                        unsigned int synAddress = shSpk[j] * 1172;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            synAddress += lid;
                            const unsigned int ipost = d_indII[synAddress];
                            atomic_add_f_global(&d_inSynII[ipost], (-8.15999999999999994e-04f));
                        }
                    }
                }
            }
        }
        
    }
    
}

)";

// Initialize the synapse update kernel(s)
void updateSynapsesProgramKernels() {
    updatePresynapticKernel = cl::Kernel(updateSynapsesProgram, "updatePresynapticKernel");
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(0, d_glbSpkCntE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(1, d_glbSpkCntI));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(2, d_glbSpkE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(3, d_glbSpkI));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(4, d_inSynEE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(5, d_inSynEI));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(6, d_inSynIE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(7, d_inSynII));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(8, d_indEE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(9, d_indEI));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(10, d_indIE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(11, d_indII));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(12, d_rowLengthEE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(13, d_rowLengthEI));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(14, d_rowLengthIE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(15, d_rowLengthII));
}

void updateSynapses(float t) {
     {
        CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(16, t));
        
        const cl::NDRange globalWorkSize(11104, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updatePresynapticKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
