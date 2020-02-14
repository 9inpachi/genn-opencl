#include "definitionsInternal.h"

void initialize() {
    unsigned long deviceRNGSeed = 0;

	initKernel.setArg(0, deviceRNGSeed);

	// Creating an initQueue for running initialization kernel
	commandQueue.enqueueNDRangeKernel(initKernel, cl::NullRange, cl::NDRange(32));
	commandQueue.finish();
}