#include "definitionsInternal.h"
#include <clRNG/philox432.h>

// Initialize kernel
extern "C" const char* initKernelSource = R"(typedef float scalar;

#define CLRNG_SINGLE_PRECISION
#include <clRNG/philox432.clh>

__kernel void initializeRNGKernel(__global clrngPhilox432HostStream *streams, unsigned int deviceRNGSeed) {
    
}

)";

// Initialize the initialization kernel
void initInitKernel() {
    
}

void initialize() {
    
}

// Initialize all OpenCL elements
void initializeSparse() {

}