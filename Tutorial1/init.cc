#include <OpenCLModule.h>
#include "definitionsInternal.h"

void initialize() {
    unsigned long deviceRNGSeed = 0;
     {
		OpenCLModule::OpenCLProgram oclProgram("init.cl", 1);

        //Buffers
		cl::Buffer b_glbSpkCntNeurons(oclProgram.context, dd_glbSpkCntNeurons.begin(), dd_glbSpkCntNeurons.end(), false);
		cl::Buffer b_glbSpkNeurons(oclProgram.context, dd_glbSpkNeurons.begin(), dd_glbSpkNeurons.end(), false);
		cl::Buffer b_VNeurons(oclProgram.context, dd_VNeurons.begin(), dd_VNeurons.end(), false);
		cl::Buffer b_UNeurons(oclProgram.context, dd_UNeurons.begin(), dd_UNeurons.end(), false);

		cl::Kernel initKernel(oclProgram.program, "initializeKernel");

		cl::CommandQueue initQueue(oclProgram.context, oclProgram.device);
		initQueue.enqueueNDRangeKernel(initKernel, cl::NullRange, cl::NDRange(32));
		
		//Reading the buffers back after execution
		initQueue.enqueueReadBuffer(b_glbSpkCntNeurons, CL_TRUE, 0, sizeof(unsigned int) * dd_glbSpkCntNeurons.size(), dd_glbSpkCntNeurons.data());
		initQueue.enqueueReadBuffer(b_glbSpkNeurons, CL_TRUE, 0, sizeof(unsigned int) * dd_glbSpkNeurons.size(), dd_glbSpkNeurons.data());
		initQueue.enqueueReadBuffer(b_VNeurons, CL_TRUE, 0, sizeof(scalar) * dd_VNeurons.size(), dd_VNeurons.data());
		initQueue.enqueueReadBuffer(b_UNeurons, CL_TRUE, 0, sizeof(scalar) * dd_UNeurons.size(), dd_UNeurons.data());

    }
}

void initializeSparse() {
    copyStateToDevice(true);
    copyConnectivityToDevice(true);
    
}