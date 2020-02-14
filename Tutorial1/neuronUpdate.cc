#include "definitionsInternal.h"

void updateNeurons(float t) {

	// preNeuronResetKernel and updateNeuronsKernel
	cl_int err = CL_SUCCESS;

	// preNeuronResetKernel

	// Creating a preNeuronResetQueue for running the preNeuronResetKernel
	err = commandQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, cl::NDRange(32));
	commandQueue.finish();

	std::string err_here = opencl::getCLError(err);

	// updateNeuronsKernel - using the same cl::Program
	
	// Setting kernel arguments
	err = updateNeuronsKernel.setArg(0, t);

	// Running the updateNeuronsKernel
	err = commandQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, cl::NDRange(32));
	commandQueue.finish();

	// Check for any errors caught during kernel execution
	std::string err_here1 = opencl::getCLError(err);
	
}