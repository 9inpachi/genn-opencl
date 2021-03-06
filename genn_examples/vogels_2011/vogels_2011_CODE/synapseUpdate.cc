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

__kernel void updatePresynapticKernel(__global scalar* d_gIE, __global unsigned int* d_glbSpkCntE, __global unsigned int* d_glbSpkCntI, __global unsigned int* d_glbSpkE, __global unsigned int* d_glbSpkI, __global float* d_inSynEE, __global float* d_inSynEI, __global float* d_inSynIE, __global float* d_inSynII, __global unsigned int* d_indEE, __global unsigned int* d_indEI, __global unsigned int* d_indIE, __global unsigned int* d_indII, __global unsigned int* d_rowLengthEE, __global unsigned int* d_rowLengthEI, __global unsigned int* d_rowLengthIE, __global unsigned int* d_rowLengthII, __global scalar* d_sTE, float t) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    __local unsigned int shRowLength[32];
    __local unsigned int shSpk[32];
    // EE
    if(id < 96) {
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
                    if (id < 77) {
                        unsigned int synAddress = shSpk[j] * 77;
                        const unsigned int npost = shRowLength[j];
                        if (id < npost) {
                            synAddress += id;
                            const unsigned int ipost = d_indEE[synAddress];
                            atomic_add_f_global(&d_inSynEE[ipost], (2.99999999999999989e-02f));
                        }
                    }
                }
            }
        }
        
    }
    
    // EI
    if(id >= 96 && id < 128) {
        const unsigned int lid = id - 96;
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
                    if (lid < 31) {
                        unsigned int synAddress = shSpk[j] * 31;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            synAddress += lid;
                            const unsigned int ipost = d_indEI[synAddress];
                            atomic_add_f_global(&d_inSynEI[ipost], (2.99999999999999989e-02f));
                        }
                    }
                }
            }
        }
        
    }
    
    // IE
    if(id >= 128 && id < 224) {
        const unsigned int lid = id - 128;
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
                    if (lid < 75) {
                        unsigned int synAddress = shSpk[j] * 75;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            synAddress += lid;
                            const unsigned int ipost = d_indIE[synAddress];
                            atomic_add_f_global(&d_inSynIE[ipost], d_gIE[synAddress]);
                            scalar dt = t - (1.00000000000000000e+00f + d_sTE[ipost]); 
                            scalar timing = expf(-dt / (2.00000000000000000e+01f)) - (1.19999999999999996e-01f);
                            scalar newWeight = d_gIE[synAddress] - ((5.00000000000000010e-03f) * timing);
                            if(newWeight < (-1.00000000000000000e+00f))
                            {
                              d_gIE[synAddress] = (-1.00000000000000000e+00f);
                            }
                            else if(newWeight > (0.00000000000000000e+00f))
                            {
                              d_gIE[synAddress] = (0.00000000000000000e+00f);
                            }
                            else
                            {
                              d_gIE[synAddress] = newWeight;
                            }
                        }
                    }
                }
            }
        }
        
    }
    
    // II
    if(id >= 224 && id < 256) {
        const unsigned int lid = id - 224;
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
                    if (lid < 29) {
                        unsigned int synAddress = shSpk[j] * 29;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            synAddress += lid;
                            const unsigned int ipost = d_indII[synAddress];
                            atomic_add_f_global(&d_inSynII[ipost], (-2.99999999999999989e-02f));
                        }
                    }
                }
            }
        }
        
    }
    
}

__kernel void updatePostsynapticKernel(__global unsigned int* d_colLengthIE, __global scalar* d_gIE, __global unsigned int* d_glbSpkCntE, __global unsigned int* d_glbSpkE, __global unsigned int* d_remapIE, __global scalar* d_sTI, float t) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    __local unsigned int shSpk[32];
    __local unsigned int shColLength[32];
    // IE
    if(id < 32) {
        const unsigned int numSpikes = d_glbSpkCntE[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (localId < numSpikesInBlock) {
                const unsigned int spk = d_glbSpkE[(r * 32) + localId];
                shSpk[localId] = spk;
                shColLength[localId] = d_colLengthIE[spk];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // only work on existing neurons
            if (id < 31) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    if (id < shColLength[j]) {
                        const unsigned int synAddress = d_remapIE[(shSpk[j] * 31) + id];
                        const unsigned int ipre = synAddress / 75;
                        scalar dt = t - (1.00000000000000000e+00f + d_sTI[ipre]);
                        scalar timing = expf(-dt / (2.00000000000000000e+01f));
                        scalar newWeight = d_gIE[synAddress] - ((5.00000000000000010e-03f) * timing);
                        d_gIE[synAddress] = (newWeight < (-1.00000000000000000e+00f)) ? (-1.00000000000000000e+00f) : newWeight;
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
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(0, d_gIE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(1, d_glbSpkCntE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(2, d_glbSpkCntI));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(3, d_glbSpkE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(4, d_glbSpkI));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(5, d_inSynEE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(6, d_inSynEI));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(7, d_inSynIE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(8, d_inSynII));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(9, d_indEE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(10, d_indEI));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(11, d_indIE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(12, d_indII));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(13, d_rowLengthEE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(14, d_rowLengthEI));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(15, d_rowLengthIE));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(16, d_rowLengthII));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(17, d_sTE));
    updatePostsynapticKernel = cl::Kernel(updateSynapsesProgram, "updatePostsynapticKernel");
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(0, d_colLengthIE));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(1, d_gIE));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(2, d_glbSpkCntE));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(3, d_glbSpkE));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(4, d_remapIE));
    CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(5, d_sTI));
}

void updateSynapses(float t) {
     {
        CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(18, t));
        
        const cl::NDRange globalWorkSize(256, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updatePresynapticKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
     {
        CHECK_OPENCL_ERRORS(updatePostsynapticKernel.setArg(6, t));
        
        const cl::NDRange globalWorkSize(32, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updatePostsynapticKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
