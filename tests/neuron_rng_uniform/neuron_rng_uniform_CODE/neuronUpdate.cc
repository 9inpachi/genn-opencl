#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateNeuronsProgramSrc = R"(typedef float scalar;

#define fmodf fmod
#define DT 0.100000f

// ------------------------------------------------------------------------
// support code
// ------------------------------------------------------------------------
#define SUPPORT_CODE_FUNC
// support code for neuron groups

// support code for synapse groups

#define CLRNG_SINGLE_PRECISION
#include <clRNG/lfsr113.clh>

__kernel void preNeuronResetKernel(__global unsigned int* d_glbSpkCntPop) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    if(id == 0) {
        d_glbSpkCntPop[0] = 0;
    }
}

__kernel void updateNeuronsKernel(__global clrngLfsr113HostStream* d_rngPop, __global scalar* d_xPop, float t) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    // Pop
    if(id < 1024) {
        
        if(id < 1000) {
            clrngLfsr113Stream localStream;
            clrngLfsr113CopyOverStreamsFromGlobal(1, &localStream, &d_rngPop[id]);
            
            scalar lx = d_xPop[id];
            
            // calculate membrane potential
            lx= clrngLfsr113RandomU01(&localStream);
            
            d_xPop[id] = lx;
            
            clrngLfsr113CopyOverStreamsToGlobal(1, &d_rngPop[id], &localStream);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
}
)";

// Initialize the neuronUpdate kernels
void updateNeuronsProgramKernels() {
    preNeuronResetKernel = cl::Kernel(updateNeuronsProgram, "preNeuronResetKernel");
    updateNeuronsProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clDevice);
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(0, d_glbSpkCntPop));
    
    updateNeuronsKernel = cl::Kernel(updateNeuronsProgram, "updateNeuronsKernel");
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(0, d_rngPop));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(1, d_xPop));
}

void updateNeurons(float t) {
     {
        const cl::NDRange globalWorkSize(32, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
        
    }
     {
        CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(2, t));
        
        const cl::NDRange globalWorkSize(1024, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
