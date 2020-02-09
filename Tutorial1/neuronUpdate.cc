#include "definitions.h"


OpenCLModule::OpenCLProgram updateNeuronsProgram("updateNeuronsKernels.cl", 1);

void updateNeurons(float t) {
	// Running the preNeuronResetKernel and updateNeuronsKernel

	cl_int err = CL_SUCCESS;

	cl::Buffer b_glbSpkCntNeurons(updateNeuronsProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 1 * sizeof(dd_glbSpkCntNeurons), dd_glbSpkCntNeurons, &err);

	cl::Kernel preNeuronResetKernel(updateNeuronsProgram.program, "preNeuronResetKernel", &err);
	err = preNeuronResetKernel.setArg(0, b_glbSpkCntNeurons);

	cl::CommandQueue preNeuronResetQueue(updateNeuronsProgram.context, updateNeuronsProgram.device);
	err = preNeuronResetQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, cl::NDRange(32));
	err = preNeuronResetQueue.enqueueReadBuffer(b_glbSpkCntNeurons, CL_TRUE, 0, sizeof(unsigned int) * 1, dd_glbSpkCntNeurons);

	std::string err_here = updateNeuronsProgram.getCLError(err);

	// updateNeuronsKernel
	// Using the same program
	cl::Buffer bu_glbSpkCntNeurons(updateNeuronsProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 1 * sizeof(dd_glbSpkCntNeurons), dd_glbSpkCntNeurons, &err);
	cl::Buffer b_glbSpkNeurons(updateNeuronsProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 1 * sizeof(dd_glbSpkNeurons), dd_glbSpkNeurons, &err);
	cl::Buffer b_VNeurons(updateNeuronsProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_VNeurons), dd_VNeurons);
	cl::Buffer b_UNeurons(updateNeuronsProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_UNeurons), dd_UNeurons);
	cl::Buffer b_aNeurons(updateNeuronsProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_aNeurons), dd_aNeurons);
	cl::Buffer b_bNeurons(updateNeuronsProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_bNeurons), dd_bNeurons);
	cl::Buffer b_cNeurons(updateNeuronsProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_cNeurons), dd_cNeurons);
	cl::Buffer b_dNeurons(updateNeuronsProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_dNeurons), dd_dNeurons);
	
	cl::Kernel updateNeuronsKernel(updateNeuronsProgram.program, "updateNeuronsKernel", &err);
	err = updateNeuronsKernel.setArg(0, t);
	err = updateNeuronsKernel.setArg(1, DT);
	err = updateNeuronsKernel.setArg(2, bu_glbSpkCntNeurons);
	err = updateNeuronsKernel.setArg(3, b_glbSpkNeurons);
	err = updateNeuronsKernel.setArg(4, b_VNeurons);
	err = updateNeuronsKernel.setArg(5, b_UNeurons);
	err = updateNeuronsKernel.setArg(6, b_aNeurons);
	err = updateNeuronsKernel.setArg(7, b_bNeurons);
	err = updateNeuronsKernel.setArg(8, b_cNeurons);
	err = updateNeuronsKernel.setArg(9, b_dNeurons);

	cl::CommandQueue updateNeuronsQueue(updateNeuronsProgram.context, updateNeuronsProgram.device);
	err = updateNeuronsQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, cl::NDRange(32));

	updateNeuronsQueue.enqueueReadBuffer(bu_glbSpkCntNeurons, CL_TRUE, 0, sizeof(unsigned int) * 1, dd_glbSpkCntNeurons);
	updateNeuronsQueue.enqueueReadBuffer(b_glbSpkNeurons, CL_TRUE, 0, sizeof(unsigned int) * 1, dd_glbSpkNeurons);
	updateNeuronsQueue.enqueueReadBuffer(b_VNeurons, CL_TRUE, 0, sizeof(scalar) * NSIZE, dd_VNeurons);
	updateNeuronsQueue.enqueueReadBuffer(b_UNeurons, CL_TRUE, 0, sizeof(scalar) * NSIZE, dd_UNeurons);
	updateNeuronsQueue.enqueueReadBuffer(b_aNeurons, CL_TRUE, 0, sizeof(scalar) * NSIZE, dd_aNeurons);
	updateNeuronsQueue.enqueueReadBuffer(b_bNeurons, CL_TRUE, 0, sizeof(scalar) * NSIZE, dd_bNeurons);
	updateNeuronsQueue.enqueueReadBuffer(b_cNeurons, CL_TRUE, 0, sizeof(scalar) * NSIZE, dd_cNeurons);
	updateNeuronsQueue.enqueueReadBuffer(b_dNeurons, CL_TRUE, 0, sizeof(scalar) * NSIZE, dd_dNeurons);

	std::string err_here1 = updateNeuronsProgram.getCLError(err);

}