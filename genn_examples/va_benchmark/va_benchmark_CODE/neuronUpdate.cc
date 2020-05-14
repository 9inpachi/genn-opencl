#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateNeuronsProgramSrc = R"(typedef float scalar;

__kernel void preNeuronResetKernel(__global unsigned int* d_glbSpkCntE, __global unsigned int* d_glbSpkCntI) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    if(id == 0) {
        d_glbSpkCntE[0] = 0;
    }
    else if(id == 1) {
        d_glbSpkCntI[0] = 0;
    }
}

__kernel void updateNeuronsKernel(const float DT, __global scalar* d_RefracTimeE, __global scalar* d_RefracTimeI, __global scalar* d_VE, __global scalar* d_VI, __global unsigned int* d_glbSpkCntE, __global unsigned int* d_glbSpkCntI, __global unsigned int* d_glbSpkE, __global unsigned int* d_glbSpkI, __global scalar* d_inSynEE, __global scalar* d_inSynEI, __global scalar* d_inSynIE, __global scalar* d_inSynII, float t) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    volatile __local unsigned int shSpk[32];
    volatile __local unsigned int shPosSpk;
    volatile __local unsigned int shSpkCount;
    if (localId == 0); {
        shSpkCount = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    // E
    if(id < 40000) {
        
        if(id < 40000) {
            scalar lV = d_VE[id];
            scalar lRefracTime = d_RefracTimeE[id];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynIE = d_inSynIE[id];
            Isyn += (9.51625819640404824e-01f) * linSynIE;
            // pull inSyn values in a coalesced access
            float linSynEE = d_inSynEE[id];
            Isyn += (9.06346234610090895e-01f) * linSynEE;
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            if (lRefracTime <= 0.0f) {
              scalar alpha = ((Isyn + (0.00000000000000000e+00f)) * (2.00000000000000000e+01f)) + (-4.90000000000000000e+01f);
              lV = alpha - ((9.51229424500714016e-01f) * (alpha - lV));
            }
            else {
              lRefracTime -= DT;
            }
            
            // test for and register a true spike
            if (lRefracTime <= 0.0f && lV >= (-5.00000000000000000e+01f)) {
                const unsigned int spkIdx = atomic_add(&shSpkCount, 1);
                shSpk[spkIdx] = id;
                // spike reset code
                lV = (-6.00000000000000000e+01f);
                lRefracTime = (5.00000000000000000e+00f);
                
            }
            d_VE[id] = lV;
            d_RefracTimeE[id] = lRefracTime;
            // the post-synaptic dynamics
            linSynIE *= (9.04837418035959518e-01f);
            d_inSynIE[id] = linSynIE;
            // the post-synaptic dynamics
            linSynEE *= (8.18730753077981821e-01f);
            d_inSynEE[id] = linSynEE;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomic_add(&d_glbSpkCntE[0], shSpkCount);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < shSpkCount) {
            const unsigned int n = shSpk[localId];
            d_glbSpkE[shPosSpk + localId] = n;
        }
    }
    
    // I
    if(id >= 40000 && id < 50016) {
        const unsigned int lid = id - 40000;
        
        if(lid < 10000) {
            scalar lV = d_VI[lid];
            scalar lRefracTime = d_RefracTimeI[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynII = d_inSynII[lid];
            Isyn += (9.51625819640404824e-01f) * linSynII;
            // pull inSyn values in a coalesced access
            float linSynEI = d_inSynEI[lid];
            Isyn += (9.06346234610090895e-01f) * linSynEI;
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            if (lRefracTime <= 0.0f) {
              scalar alpha = ((Isyn + (0.00000000000000000e+00f)) * (2.00000000000000000e+01f)) + (-4.90000000000000000e+01f);
              lV = alpha - ((9.51229424500714016e-01f) * (alpha - lV));
            }
            else {
              lRefracTime -= DT;
            }
            
            // test for and register a true spike
            if (lRefracTime <= 0.0f && lV >= (-5.00000000000000000e+01f)) {
                const unsigned int spkIdx = atomic_add(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                // spike reset code
                lV = (-6.00000000000000000e+01f);
                lRefracTime = (5.00000000000000000e+00f);
                
            }
            d_VI[lid] = lV;
            d_RefracTimeI[lid] = lRefracTime;
            // the post-synaptic dynamics
            linSynII *= (9.04837418035959518e-01f);
            d_inSynII[lid] = linSynII;
            // the post-synaptic dynamics
            linSynEI *= (8.18730753077981821e-01f);
            d_inSynEI[lid] = linSynEI;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomic_add(&d_glbSpkCntI[0], shSpkCount);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < shSpkCount) {
            const unsigned int n = shSpk[localId];
            d_glbSpkI[shPosSpk + localId] = n;
        }
    }
    
}
)";

// Initialize the neuronUpdate kernels
void updateNeuronsProgramKernels() {
    preNeuronResetKernel = cl::Kernel(updateNeuronsProgram, "preNeuronResetKernel");
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(0, d_glbSpkCntE));
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(1, d_glbSpkCntI));
    
    updateNeuronsKernel = cl::Kernel(updateNeuronsProgram, "updateNeuronsKernel");
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(0, DT));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(1, d_RefracTimeE));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(2, d_RefracTimeI));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(3, d_VE));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(4, d_VI));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(5, d_glbSpkCntE));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(6, d_glbSpkCntI));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(7, d_glbSpkE));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(8, d_glbSpkI));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(9, d_inSynEE));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(10, d_inSynEI));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(11, d_inSynIE));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(12, d_inSynII));
}

void updateNeurons(float t) {
     {
        const cl::NDRange globalWorkSize(32, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
     {
        CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(13, t));
        
        const cl::NDRange globalWorkSize(50016, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
