#include "definitionsInternal.h"

void updateNeurons(float t) {

	// preNeuronResetKernel and updateNeuronsKernel
	cl_int err = CL_SUCCESS;

	// Buffers for updateNeuronsKernels
	b_glbSpkCntNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 1 * sizeof(unsigned int), dd_glbSpkCntNeurons);
	b_glbSpkNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(unsigned int), dd_glbSpkNeurons);
	b_VNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(scalar), dd_VNeurons);
	b_UNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(scalar), dd_UNeurons);
	b_aNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(scalar), dd_aNeurons);
	b_bNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(scalar), dd_bNeurons);
	b_cNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(scalar), dd_cNeurons);
	b_dNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(scalar), dd_dNeurons);

	// preNeuronResetKernel

	// preNeuronResetKernel to reset values
	cl::Kernel preNeuronResetKernel(unProgram, "preNeuronResetKernel", &err);
	err = preNeuronResetKernel.setArg(0, b_glbSpkCntNeurons);

	// Creating a preNeuronResetQueue for running the preNeuronResetKernel
	cl::CommandQueue preNeuronResetQueue(unContext, unDevice);
	err = preNeuronResetQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, cl::NDRange(32));

	std::string err_here = opencl::getCLError(err);

	// updateNeuronsKernel - using the same cl::Program
	
	// updateNeuronsKernel for updating neurons
	cl::Kernel updateNeuronsKernel(unProgram, "updateNeuronsKernel", &err);
	// Setting kernel arguments
	err = updateNeuronsKernel.setArg(0, t);
	err = updateNeuronsKernel.setArg(1, DT);
	err = updateNeuronsKernel.setArg(2, b_glbSpkCntNeurons);
	err = updateNeuronsKernel.setArg(3, b_glbSpkNeurons);
	err = updateNeuronsKernel.setArg(4, b_VNeurons);
	err = updateNeuronsKernel.setArg(5, b_UNeurons);
	err = updateNeuronsKernel.setArg(6, b_aNeurons);
	err = updateNeuronsKernel.setArg(7, b_bNeurons);
	err = updateNeuronsKernel.setArg(8, b_cNeurons);
	err = updateNeuronsKernel.setArg(9, b_dNeurons);

	// Creating an updateNeuronsQueue to run the updateNeuronsKernel
	cl::CommandQueue updateNeuronsQueue(unContext, unDevice);
	err = updateNeuronsQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, cl::NDRange(32));

	updateNeuronsQueue.finish();

	// Check for any errors caught during kernel execution
	std::string err_here1 = opencl::getCLError(err);
	
}