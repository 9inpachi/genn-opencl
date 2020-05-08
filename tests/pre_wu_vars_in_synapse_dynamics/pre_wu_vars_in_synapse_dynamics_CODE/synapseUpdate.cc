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

void atomic_add_f_local(volatile __local float *source, const float operand) {
    union { unsigned int intVal; float floatVal; } newVal;
    union { unsigned int intVal; float floatVal; } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    }
    while (atomic_cmpxchg((volatile __local unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void updateSynapseDynamicsKernel(__global float* d_inSynsyn, __global unsigned int* d_indsyn, __global scalar* d_ssyn, __global unsigned int* d_synRemapsyn, __global scalar* d_wsyn, volatile unsigned int spkQuePtrpre, float t) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    // syn
    if(id < 32) {
        const unsigned int preReadDelayOffset = ((spkQuePtrpre + 1) % 21) * 10;
        if (id < d_synRemapsyn[0]) {
            const unsigned int s = d_synRemapsyn[1 + id];
            d_wsyn[s]= d_ssyn[preReadDelayOffset + s / 1];
        }
    }
    
}

)";

// Initialize the synapse update kernel(s)
void updateSynapsesProgramKernels() {
    updateSynapseDynamicsKernel = cl::Kernel(updateSynapsesProgram, "updateSynapseDynamicsKernel");
    CHECK_OPENCL_ERRORS(updateSynapseDynamicsKernel.setArg(0, d_inSynsyn));
    CHECK_OPENCL_ERRORS(updateSynapseDynamicsKernel.setArg(1, d_indsyn));
    CHECK_OPENCL_ERRORS(updateSynapseDynamicsKernel.setArg(2, d_ssyn));
    CHECK_OPENCL_ERRORS(updateSynapseDynamicsKernel.setArg(3, d_synRemapsyn));
    CHECK_OPENCL_ERRORS(updateSynapseDynamicsKernel.setArg(4, d_wsyn));
}

void updateSynapses(float t) {
     {
        CHECK_OPENCL_ERRORS(updateSynapseDynamicsKernel.setArg(5, spkQuePtrpre));
        CHECK_OPENCL_ERRORS(updateSynapseDynamicsKernel.setArg(6, t));
        
        const cl::NDRange globalWorkSize(32, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updateSynapseDynamicsKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
