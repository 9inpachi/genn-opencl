#include "definitionsInternal.h"

void initialize() {
    unsigned long deviceRNGSeed = 0;

	cl_int err = CL_SUCCESS;

	// initKernel to initialize values
	cl::Kernel initKernel(initProgram, "initializeKernel", &err);
	// Setting kernel arguments
	err = initKernel.setArg(0, deviceRNGSeed);
	err = initKernel.setArg(1, b_glbSpkCntNeurons);
	err = initKernel.setArg(2, b_glbSpkNeurons);
	err = initKernel.setArg(3, b_VNeurons);
	err = initKernel.setArg(4, b_UNeurons);

	// Creating an initQueue for running initialization kernel
	err = commandQueue.enqueueNDRangeKernel(initKernel, cl::NullRange, cl::NDRange(32));
	commandQueue.finish();

	// Catching any errors in kernels
	std::string err_here_init = opencl::getCLError(err);
}