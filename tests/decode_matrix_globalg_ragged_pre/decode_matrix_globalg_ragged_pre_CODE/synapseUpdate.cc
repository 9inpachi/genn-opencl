#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateSynapsesProgramSrc = R"(typedef float scalar;

void atomic_add_f(volatile __local float *source, const float operand) {
    union { unsigned int intVal; float floatVal; } newVal;
    union { unsigned int intVal; float floatVal; } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    }
    while (atomic_cmpxchg((volatile __local unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void updatePresynapticKernel(__global unsigned int* d_glbSpkCntPre, __global unsigned int* d_glbSpkPre, __global unsigned int* d_inSynSyn, __global unsigned int* d_indSyn, __global unsigned int* d_rowLengthSyn, float t) {
    size_t groupId = get_group_id(0);
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    __local float shLg[32];
    __local unsigned int shSpk[32];
    // Syn
    if(id < 32) {
        if(localId < 4) {
            shLg[localId] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
         {
            const unsigned int spike = id;
            if (spike < d_glbSpkCntPre[0]) {
                const unsigned int preInd = d_glbSpkPre[spike];
                unsigned int synAddress = preInd * 4;
                const unsigned int npost = d_rowLengthSyn[preInd];
                for(unsigned int i = 0; i < npost; i++, synAddress++) {
                    const unsigned int ipost = d_indSyn[synAddress];
                    atomic_add_f(&shLg[ipost], (1.00000000000000000e+00f));
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < 4) {
            d_inSynSyn[localId] += shLg[localId];

            printf("shLg[%d] = %f\n", localId, shLg[localId]);
            printf("d_inSynSyn[%d] = %f\n", localId, d_inSynSyn[localId]);
            printf("d_inSynSyn[%d] (%f) + shLg[%d] (%f) = %f\n", localId, d_inSynSyn[localId], localId, shLg[localId], d_inSynSyn[localId] + shLg[localId]);
        }
    }
    
}

)";

// Initialize the synapse update kernel(s)
void updateSynapsesProgramKernels() {
    updatePresynapticKernel = cl::Kernel(updateSynapsesProgram, "updatePresynapticKernel");
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(0, d_glbSpkCntPre));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(1, d_glbSpkPre));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(2, d_inSynSyn));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(3, d_indSyn));
    CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(4, d_rowLengthSyn));
}

void updateSynapses(float) {
     {
        const cl::NDRange global(32, 1);
        const cl::NDRange local(32, 1);
        CHECK_OPENCL_ERRORS(updatePresynapticKernel.setArg(5, t));
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updatePresynapticKernel, cl::NDRange(0), global, local));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
