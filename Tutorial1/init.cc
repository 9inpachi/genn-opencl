#include "definitionsInternal.h"

void initialize() {
    unsigned long deviceRNGSeed = 0;

	cl_int err = CL_SUCCESS;

	// Buffers for initKernel
	b_init_glbSpkCntNeurons = cl::Buffer(initContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 1 * sizeof(unsigned int), dd_glbSpkCntNeurons);
	b_init_glbSpkNeurons = cl::Buffer(initContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(unsigned int), dd_glbSpkNeurons);
	b_init_VNeurons = cl::Buffer(initContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(scalar), dd_VNeurons);
	b_init_UNeurons = cl::Buffer(initContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(scalar), dd_UNeurons);

	// initKernel to initialize values
	cl::Kernel initKernel(initProgram, "initializeKernel", &err);
	// Setting kernel arguments
	err = initKernel.setArg(0, deviceRNGSeed);
	err = initKernel.setArg(1, b_init_glbSpkCntNeurons);
	err = initKernel.setArg(2, b_init_glbSpkNeurons);
	err = initKernel.setArg(3, b_init_VNeurons);
	err = initKernel.setArg(4, b_init_UNeurons);

	// Creating an initQueue for running initialization kernel
	cl::CommandQueue initQueue(initContext, initDevice);
	err = initQueue.enqueueNDRangeKernel(initKernel, cl::NullRange, cl::NDRange(32));

	initQueue.finish();

	// Catching any errors in kernels
	std::string err_here_init = opencl::getCLError(err);
}