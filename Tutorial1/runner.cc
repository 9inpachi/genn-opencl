#include "definitionsInternal.h"

extern "C" {
	unsigned int* glbSpkCntNeurons;
	unsigned int* glbSpkNeurons;
	scalar* VNeurons;
	scalar* UNeurons;
	scalar* aNeurons;
	scalar* bNeurons;
	scalar* cNeurons;
	scalar* dNeurons;
	unsigned long long iT;
	float t;

	// Buffers
	cl::Buffer db_glbSpkCntNeurons;
	cl::Buffer db_glbSpkNeurons;
	cl::Buffer db_VNeurons;
	cl::Buffer db_UNeurons;
	cl::Buffer db_aNeurons;
	cl::Buffer db_bNeurons;
	cl::Buffer db_cNeurons;
	cl::Buffer db_dNeurons;

	// OpenCL variables
	cl::Context clContext;
	cl::Device clDevice;
	cl::Program initProgram;
	cl::Program unProgram;
	cl::CommandQueue commandQueue;

	// OpenCL kernels
	cl::Kernel initKernel;
	cl::Kernel preNeuronResetKernel;
	cl::Kernel updateNeuronsKernel;
}

// Allocating memory to pointers
void allocateMem() {
	// Allocating memory to host pointers
	glbSpkCntNeurons = (unsigned int*)malloc(1 * sizeof(unsigned int));
	glbSpkNeurons = (unsigned int*)malloc(NSIZE * sizeof(unsigned int));
	VNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	UNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	aNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	bNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	cNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	dNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
}

// Initializing kernel programs so that they can be used to run the kernels
void initKernelPrograms() {
	opencl::setUpContext(clContext, clDevice, DEVICE_INDEX);
	opencl::createProgram("init.cl", initProgram, clContext);
	opencl::createProgram("updateNeuronsKernels.cl", unProgram, clContext);
	commandQueue = cl::CommandQueue(clContext, clDevice);
}

// Initialize buffers to be used by OpenCL kernels
void initBuffers() {
	// Buffers
	db_glbSpkCntNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof(glbSpkCntNeurons), glbSpkCntNeurons);
	db_glbSpkNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NSIZE * sizeof(glbSpkNeurons), glbSpkNeurons);
	db_VNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NSIZE * sizeof(VNeurons), VNeurons);
	db_UNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NSIZE * sizeof(UNeurons), UNeurons);
	db_aNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NSIZE * sizeof(aNeurons), aNeurons);
	db_bNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NSIZE * sizeof(bNeurons), bNeurons);
	db_cNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NSIZE * sizeof(cNeurons), cNeurons);
	db_dNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NSIZE * sizeof(dNeurons), dNeurons);
}

// Initialize kernels and set their arguments so they can directly run with the commandQueue
void initKernels() {
	cl_int err = CL_SUCCESS;

	initKernel = cl::Kernel(initProgram, "initializeKernel");
	// Setting kernel arguments
	err = initKernel.setArg(1, db_glbSpkCntNeurons);
	err = initKernel.setArg(2, db_glbSpkNeurons);
	err = initKernel.setArg(3, db_VNeurons);
	err = initKernel.setArg(4, db_UNeurons);

	preNeuronResetKernel = cl::Kernel(unProgram, "preNeuronResetKernel");
	err = preNeuronResetKernel.setArg(0, db_glbSpkCntNeurons);

	updateNeuronsKernel = cl::Kernel(unProgram, "updateNeuronsKernel");
	err = updateNeuronsKernel.setArg(1, DT);
	err = updateNeuronsKernel.setArg(2, db_glbSpkCntNeurons);
	err = updateNeuronsKernel.setArg(3, db_glbSpkNeurons);
	err = updateNeuronsKernel.setArg(4, db_VNeurons);
	err = updateNeuronsKernel.setArg(5, db_UNeurons);
	err = updateNeuronsKernel.setArg(6, db_aNeurons);
	err = updateNeuronsKernel.setArg(7, db_bNeurons);
	err = updateNeuronsKernel.setArg(8, db_cNeurons);
	err = updateNeuronsKernel.setArg(9, db_dNeurons);

	std::string errStr = opencl::getCLError(err);
}

void initializeOpenCL() {
	initKernelPrograms();
	initBuffers();
	initKernels();
}

void stepTime() {
	updateNeurons(t);
	iT++;
	t = iT * DT;
}

scalar* getCurrentVNeurons() {
	return VNeurons;
}

void pullCurrentVNeuronsFromDevice() {
	commandQueue.enqueueReadBuffer(db_VNeurons, CL_TRUE, 0, NSIZE * sizeof(scalar), VNeurons);
}