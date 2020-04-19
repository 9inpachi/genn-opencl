#include "definitionsInternal.h"
#include <clRNG/philox432.h>

// Initialize kernel
extern "C" const char* initKernelSource = R"(typedef float scalar;

__kernel void initializeRNGKernel(
    unsigned int deviceRNGSeed
) {

}

__kernel void initializeKernel(
    __global unsigned int* dd_glbSpkCntExc,
    __global unsigned int* dd_glbSpkExc,
    __global scalar* dd_VExc,
    __global scalar* dd_UExc, 
    __global float* dd_inSynInh_Exc,
    __global float* dd_inSynExc_Exc,
    __global unsigned int* dd_rowLengthExc_Exc,
    __global unsigned int* dd_indExc_Exc,
    __global unsigned int* dd_rowLengthInh_Exc,
    __global unsigned int* dd_indInh_Exc,
    __global unsigned int* dd_glbSpkCntInh,
    __global unsigned int* dd_glbSpkInh,
    __global scalar* dd_VInh,
    __global scalar* dd_UInh, 
    __global float* dd_inSynInh_Inh,
    __global float* dd_inSynExc_Inh,
    __global unsigned int* dd_rowLengthExc_Inh,
    __global unsigned int* dd_indExc_Inh,
    __global unsigned int* dd_rowLengthInh_Inh,
    __global unsigned int* dd_indInh_Inh,
    unsigned int deviceRNGSeed
) {
    size_t groupId = get_group_id(0);
    size_t localId = get_local_id(0);
    const unsigned int id = 64 * groupId + localId;
    // ------------------------------------------------------------------------
    // Remote neuron groups
    
    // ------------------------------------------------------------------------
    // Local neuron groups
    // Exc
    if(id < 8000) {
        // only do this for existing neurons
        if(id < 8000) {
            if(id == 0) {
                dd_glbSpkCntExc[0] = 0;
            }
            dd_glbSpkExc[id] = 0;
             {
                dd_VExc[id] = (-6.50000000000000000e+01f);
            }
             {
                const scalar scale = (3.00000000000000000e+01f) - (0.00000000000000000e+00f);
                dd_UExc[id] = (0.00000000000000000e+00f) + (1 * scale);
            }
            dd_inSynInh_Exc[id] = 0.000000f;
            dd_inSynExc_Exc[id] = 0.000000f;
            // current source variables
        }
    }
    
    // Inh
    if(id >= 8000 && id < 10048) {
        const unsigned int lid = id - 8000;
        // only do this for existing neurons
        if(lid < 2000) {
            if(lid == 0) {
                dd_glbSpkCntInh[0] = 0;
            }
            dd_glbSpkInh[lid] = 0;
             {
                dd_VInh[lid] = (-6.50000000000000000e+01f);
            }
             {
                const scalar scale = (3.00000000000000000e+01f) - (0.00000000000000000e+00f);
                dd_UInh[lid] = (0.00000000000000000e+00f) + (1 * scale);
            }
            dd_inSynInh_Inh[lid] = 0.000000f;
            dd_inSynExc_Inh[lid] = 0.000000f;
            // current source variables
        }
    }
    
    
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
    // Exc_Exc
    if(id >= 10048 && id < 18048) {
        const unsigned int lid = id - 10048;
        // only do this for existing presynaptic neurons
        if(lid < 8000) {
            dd_rowLengthExc_Exc[lid] = 0;
            // Build sparse connectivity
            int prevJ = -1;
            while(true) {
                int nextJ;
                do {
                   const scalar u = 1.0;
                   nextJ = prevJ + (1 + (int)(log(u) * (-9.49122158102990454e+00f)));
                } while(nextJ == lid);
                prevJ = nextJ;
                if(prevJ < 8000) {
                   dd_indExc_Exc[(lid * 953) + (dd_rowLengthExc_Exc[lid]++)] = prevJ;
                }
                else {
                   break;
                }
                
            }
        }
    }
    
    // Exc_Inh
    if(id >= 18048 && id < 26048) {
        const unsigned int lid = id - 18048;
        // only do this for existing presynaptic neurons
        if(lid < 8000) {
            dd_rowLengthExc_Inh[lid] = 0;
            // Build sparse connectivity
            int prevJ = -1;
            while(true) {
                int nextJ;
                do {
                   const scalar u = 1.0;
                   nextJ = prevJ + (1 + (int)(log(u) * (-9.49122158102990454e+00f)));
                } while(nextJ == lid);
                prevJ = nextJ;
                if(prevJ < 2000) {
                   dd_indExc_Inh[(lid * 279) + (dd_rowLengthExc_Inh[lid]++)] = prevJ;
                }
                else {
                   break;
                }
                
            }
        }
    }
    
    // Inh_Exc
    if(id >= 26048 && id < 28096) {
        const unsigned int lid = id - 26048;
        // only do this for existing presynaptic neurons
        if(lid < 2000) {
            dd_rowLengthInh_Exc[lid] = 0;
            // Build sparse connectivity
            int prevJ = -1;
            while(true) {
                int nextJ;
                do {
                   const scalar u = 1.0;
                   nextJ = prevJ + (1 + (int)(log(u) * (-9.49122158102990454e+00f)));
                } while(nextJ == lid);
                prevJ = nextJ;
                if(prevJ < 8000) {
                   dd_indInh_Exc[(lid * 946) + (dd_rowLengthInh_Exc[lid]++)] = prevJ;
                }
                else {
                   break;
                }
                
            }
        }
    }
    
    // Inh_Inh
    if(id >= 28096 && id < 30144) {
        const unsigned int lid = id - 28096;
        // only do this for existing presynaptic neurons
        if(lid < 2000) {
            dd_rowLengthInh_Inh[lid] = 0;
            // Build sparse connectivity
            int prevJ = -1;
            while(true) {
                int nextJ;
                do {
                   const scalar u = 1.0;
                   nextJ = prevJ + (1 + (int)(log(u) * (-9.49122158102990454e+00f)));
                } while(nextJ == lid);
                prevJ = nextJ;
                if(prevJ < 2000) {
                   dd_indInh_Inh[(lid * 275) + (dd_rowLengthInh_Inh[lid]++)] = prevJ;
                }
                else {
                   break;
                }
                
            }
        }
    }
}

)";

// Initialize the initialization kernel
void initInitializationKernels() {
    initializeKernel = cl::Kernel(initProgram, "initializeKernel");
    cl_int err;
    err = initializeKernel.setArg(0, d_glbSpkCntExc);
    initializeKernel.setArg(1, d_glbSpkExc);
    initializeKernel.setArg(2, d_VExc);
    initializeKernel.setArg(3, d_UExc);
    initializeKernel.setArg(4, d_inSynInh_Exc);
    initializeKernel.setArg(5, d_inSynExc_Exc);
    initializeKernel.setArg(6, d_rowLengthExc_Exc);
    initializeKernel.setArg(7, d_indExc_Exc);
    initializeKernel.setArg(8, d_rowLengthInh_Exc);
    initializeKernel.setArg(9, d_indInh_Exc);
    initializeKernel.setArg(10, d_glbSpkCntInh);
    initializeKernel.setArg(11, d_glbSpkInh);
    initializeKernel.setArg(12, d_VInh);
    initializeKernel.setArg(13, d_UInh);
    initializeKernel.setArg(14, d_inSynInh_Inh);
    initializeKernel.setArg(15, d_inSynExc_Inh);
    initializeKernel.setArg(16, d_rowLengthExc_Inh);
    initializeKernel.setArg(17, d_indExc_Inh);
    initializeKernel.setArg(18, d_rowLengthInh_Inh);
    initializeKernel.setArg(19, d_indInh_Inh);
}

void initialize() {
    unsigned int deviceRNGSeed = 0;
    initializeKernel.setArg(20, deviceRNGSeed);

    // 471 - global size | 64 - local size
    commandQueue.enqueueNDRangeKernel(initializeKernel, cl::NullRange, cl::NDRange(471), cl::NDRange(64));
    commandQueue.finish();
}

// Initialize all OpenCL elements
void initializeSparse() {
    copyStateToDevice(true);
}