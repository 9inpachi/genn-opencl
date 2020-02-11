#include "definitionsInternal.h"

void updateNeurons(float t) {

	// preNeuronResetKernel and updateNeuronsKernel
	cl_int err = CL_SUCCESS;

	// preNeuronResetKernel

	// preNeuronResetKernel to reset values
	cl::Kernel preNeuronResetKernel(unProgram, "preNeuronResetKernel", &err);
	err = preNeuronResetKernel.setArg(0, b_glbSpkCntNeurons);

	// Creating a preNeuronResetQueue for running the preNeuronResetKernel
	cl::CommandQueue preNeuronResetQueue(unContext, unDevice);
	err = preNeuronResetQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, cl::NDRange(32));
	preNeuronResetQueue.finish();

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