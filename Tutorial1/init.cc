#include "definitions.h"

void initialize() {
    unsigned long deviceRNGSeed = 0;
     {
		OpenCLModule::OpenCLProgram oclProgram("init.cl", 1);

		cl_int err = CL_SUCCESS;

        //Buffers
		cl::Buffer b_glbSpkCntNeurons(oclProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 1 * sizeof(dd_glbSpkCntNeurons), dd_glbSpkCntNeurons);
		cl::Buffer b_glbSpkNeurons(oclProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_glbSpkNeurons), dd_glbSpkNeurons);
		cl::Buffer b_VNeurons(oclProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_VNeurons), dd_VNeurons);
		cl::Buffer b_UNeurons(oclProgram.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_UNeurons), dd_UNeurons);

		cl::Kernel initKernel(oclProgram.program, "initializeKernel", &err);
		err = initKernel.setArg(0, deviceRNGSeed);
		err = initKernel.setArg(1, b_glbSpkCntNeurons);
		err = initKernel.setArg(2, b_glbSpkNeurons);
		err = initKernel.setArg(3, b_VNeurons);
		err = initKernel.setArg(4, b_UNeurons);

		cl::CommandQueue initQueue(oclProgram.context, oclProgram.device);
		err = initQueue.enqueueNDRangeKernel(initKernel, cl::NullRange, cl::NDRange(32));
		
		//Reading the buffers back after execution
		initQueue.enqueueReadBuffer(b_glbSpkCntNeurons, CL_TRUE, 0, sizeof(unsigned int) * NSIZE, dd_glbSpkCntNeurons);
		initQueue.enqueueReadBuffer(b_glbSpkNeurons, CL_TRUE, 0, sizeof(unsigned int) * NSIZE, dd_glbSpkNeurons);
		initQueue.enqueueReadBuffer(b_VNeurons, CL_TRUE, 0, sizeof(scalar) * NSIZE, dd_VNeurons);
		initQueue.enqueueReadBuffer(b_UNeurons, CL_TRUE, 0, sizeof(scalar) * NSIZE, dd_UNeurons);

		std::string err_here_init = oclProgram.getCLError(err);

    }
}