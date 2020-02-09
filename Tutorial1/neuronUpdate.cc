#include <OpenCLModule.h>
#include "definitions.h"

void updateNeurons(float t) {
	// Running the preNeuronResetKernel and updateNeuronsKernel

	OpenCLModule::OpenCLProgram updateNeuronsProgram("updateNeuronsKernels.cl", 1);

	cl::Buffer b_glbSpkCntNeurons(updateNeuronsProgram.context, dd_glbSpkCntNeurons.begin(), dd_glbSpkCntNeurons.end(), false);

	cl::Kernel preNeuronResetKernel(updateNeuronsProgram.program, "preNeuronResetKernel");
	preNeuronResetKernel.setArg(0, b_glbSpkCntNeurons);

	cl::CommandQueue preNeuronResetQueue(updateNeuronsProgram.context, updateNeuronsProgram.device);
	preNeuronResetQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, cl::NDRange(32));
	preNeuronResetQueue.enqueueReadBuffer(b_glbSpkCntNeurons, CL_TRUE, 0, sizeof(unsigned int) * dd_glbSpkCntNeurons.size(), dd_glbSpkCntNeurons.data());

	// updateNeuronsKernel
	// Using the same program
	cl::Buffer bu_glbSpkCntNeurons(updateNeuronsProgram.context, dd_glbSpkCntNeurons.begin(), dd_glbSpkCntNeurons.end(), false);
	cl::Buffer b_VNeurons(updateNeuronsProgram.context, dd_VNeurons.begin(), dd_VNeurons.end(), false);
	cl::Buffer b_UNeurons(updateNeuronsProgram.context, dd_UNeurons.begin(), dd_UNeurons.end(), false);
	cl::Buffer b_aNeurons(updateNeuronsProgram.context, dd_aNeurons.begin(), dd_aNeurons.end(), false);
	cl::Buffer b_bNeurons(updateNeuronsProgram.context, dd_bNeurons.begin(), dd_bNeurons.end(), false);
	cl::Buffer b_cNeurons(updateNeuronsProgram.context, dd_cNeurons.begin(), dd_cNeurons.end(), false);
	cl::Buffer b_dNeurons(updateNeuronsProgram.context, dd_dNeurons.begin(), dd_dNeurons.end(), false);
	
	cl::Kernel updateNeuronsKernel(updateNeuronsProgram.program, "preNeuronResetKernel");
	updateNeuronsKernel.setArg(0, t);
	updateNeuronsKernel.setArg(1, DT);
	updateNeuronsKernel.setArg(2, b_glbSpkCntNeurons);
	updateNeuronsKernel.setArg(3, b_VNeurons);
	updateNeuronsKernel.setArg(4, b_UNeurons);
	updateNeuronsKernel.setArg(5, b_aNeurons);
	updateNeuronsKernel.setArg(6, b_bNeurons);
	updateNeuronsKernel.setArg(7, b_cNeurons);
	updateNeuronsKernel.setArg(8, b_dNeurons);

	cl::CommandQueue updateNeuronsQueue(updateNeuronsProgram.context, updateNeuronsProgram.device);
	updateNeuronsQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, cl::NDRange(32));

	updateNeuronsQueue.enqueueReadBuffer(bu_glbSpkCntNeurons, CL_TRUE, 0, sizeof(unsigned int) * dd_glbSpkCntNeurons.size(), dd_glbSpkCntNeurons.data());
	updateNeuronsQueue.enqueueReadBuffer(b_VNeurons, CL_TRUE, 0, sizeof(unsigned int) * dd_VNeurons.size(), dd_VNeurons.data());
	updateNeuronsQueue.enqueueReadBuffer(b_UNeurons, CL_TRUE, 0, sizeof(unsigned int) * dd_UNeurons.size(), dd_UNeurons.data());
	updateNeuronsQueue.enqueueReadBuffer(b_aNeurons, CL_TRUE, 0, sizeof(unsigned int) * dd_aNeurons.size(), dd_aNeurons.data());
	updateNeuronsQueue.enqueueReadBuffer(b_bNeurons, CL_TRUE, 0, sizeof(unsigned int) * dd_bNeurons.size(), dd_bNeurons.data());
	updateNeuronsQueue.enqueueReadBuffer(b_cNeurons, CL_TRUE, 0, sizeof(unsigned int) * dd_cNeurons.size(), dd_cNeurons.data());
	updateNeuronsQueue.enqueueReadBuffer(b_dNeurons, CL_TRUE, 0, sizeof(unsigned int) * dd_dNeurons.size(), dd_dNeurons.data());


}