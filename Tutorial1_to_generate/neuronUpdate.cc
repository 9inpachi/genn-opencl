#include "definitionsInternal.h"

void updateNeurons(float t) {

	// preNeuronResetKernel

	// Creating a preNeuronResetQueue for running the preNeuronResetKernel
	commandQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, cl::NDRange(32));
	commandQueue.finish();

	// updateNeuronsKernel
	
	// Setting kernel arguments
	updateNeuronsKernel.setArg(0, t);

	// Running the updateNeuronsKernel
	commandQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, cl::NDRange(32));
	commandQueue.finish();
	
}