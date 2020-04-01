#include "definitionsInternal.h"

// Initialize kernel
extern "C" const char* initKernelSource = R"(typedef float scalar;

// Will have to read the arguments again and update for the program
__kernel void initializeKernel(const unsigned int deviceRNGSeed,
__global unsigned int* glbSpkCntNeurons,
__global unsigned int* glbSpkNeurons,
__global scalar* VNeurons,
__global scalar* UNeurons){
	int groupId = get_group_id(0);
	int localId = get_local_id(0);
	const unsigned int id = 32 * groupId + localId;
    
	if(id < 32) {
		// only do this for existing neurons
		if(id < 7) {
			if(id == 0) {
				glbSpkCntNeurons[0] = 0;
			}
			glbSpkNeurons[id] = 0;
			VNeurons[id] = (-6.50000000000000000e+01f);
			UNeurons[id] = (-2.00000000000000000e+01f);
			// current source variables
		}
	}
})";

// Initialize the initialization kernel
void initInitKernel() {
	initKernel = cl::Kernel(initProgram, "initializeKernel");
	// Setting kernel arguments
	initKernel.setArg(1, db_glbSpkCntNeurons);
	initKernel.setArg(2, db_glbSpkNeurons);
	initKernel.setArg(3, db_VNeurons);
	initKernel.setArg(4, db_UNeurons);
}

void initialize() {
    unsigned long deviceRNGSeed = 0;

	initKernel.setArg(0, deviceRNGSeed);

	// Creating an initQueue for running initialization kernel
	commandQueue.enqueueNDRangeKernel(initKernel, cl::NullRange, cl::NDRange(32));
	commandQueue.finish();
}

// Initialize all OpenCL elements
void initializeSparse() {
	copyStateToDevice(true);
}