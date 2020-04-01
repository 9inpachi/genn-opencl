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

// Initializing kernel programs so that they can be used to run the kernels
void initKernelPrograms() {
	opencl::setUpContext(clContext, clDevice, DEVICE_INDEX);
	// Create programs for kernels
	opencl::createProgram(initKernelSource, initProgram, clContext);
	opencl::createProgram(updateNeuronsKernelSource, unProgram, clContext);
	commandQueue = cl::CommandQueue(clContext, clDevice);
}

// Allocating memory to pointers
void allocateMem() {
	initKernelPrograms();
	// Allocating memory to host pointers
	glbSpkCntNeurons = (unsigned int*)malloc(1 * sizeof(unsigned int));
	glbSpkNeurons = (unsigned int*)malloc(NSIZE * sizeof(unsigned int));
	VNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	UNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	aNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	bNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	cNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	dNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));

	// Initialize buffers to be used by OpenCL kernels
	db_glbSpkCntNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE, 1 * sizeof(glbSpkCntNeurons), glbSpkCntNeurons);
	db_glbSpkNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE, NSIZE * sizeof(glbSpkNeurons), glbSpkNeurons);
	db_VNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE, NSIZE * sizeof(VNeurons), VNeurons);
	db_UNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE, NSIZE * sizeof(UNeurons), UNeurons);
	db_aNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE, NSIZE * sizeof(aNeurons), aNeurons);
	db_bNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE, NSIZE * sizeof(bNeurons), bNeurons);
	db_cNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE, NSIZE * sizeof(cNeurons), cNeurons);
	db_dNeurons = cl::Buffer(clContext, CL_MEM_READ_WRITE, NSIZE * sizeof(dNeurons), dNeurons);

	// Initializing kernels
	initInitKernel();
	initUpdateNeuronsKernels();
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

// Push functions

void pushVNeuronsToDevice(bool uninitialisedOnly) {
	commandQueue.enqueueWriteBuffer(db_VNeurons, CL_TRUE, 0, 7 * sizeof(scalar), VNeurons);
}

void pushUNeuronsToDevice(bool uninitialisedOnly) {
	commandQueue.enqueueWriteBuffer(db_UNeurons, CL_TRUE, 0, 7 * sizeof(scalar), UNeurons);
}

void pushaNeuronsToDevice(bool uninitialisedOnly) {
	commandQueue.enqueueWriteBuffer(db_aNeurons, CL_TRUE, 0, 7 * sizeof(scalar), aNeurons);
}

void pushbNeuronsToDevice(bool uninitialisedOnly) {
	commandQueue.enqueueWriteBuffer(db_bNeurons, CL_TRUE, 0, 7 * sizeof(scalar), bNeurons);
}

void pushcNeuronsToDevice(bool uninitialisedOnly) {
	commandQueue.enqueueWriteBuffer(db_cNeurons, CL_TRUE, 0, 7 * sizeof(scalar), cNeurons);
}

void pushdNeuronsToDevice(bool uninitialisedOnly) {
	commandQueue.enqueueWriteBuffer(db_dNeurons, CL_TRUE, 0, 7 * sizeof(scalar), dNeurons);
}

void pushNeuronsStateToDevice(bool uninitialisedOnly) {
	pushVNeuronsToDevice(uninitialisedOnly);
	pushUNeuronsToDevice(uninitialisedOnly);
	pushaNeuronsToDevice(uninitialisedOnly);
	pushbNeuronsToDevice(uninitialisedOnly);
	pushcNeuronsToDevice(uninitialisedOnly);
	pushdNeuronsToDevice(uninitialisedOnly);
}

void copyStateToDevice(bool uninitialisedOnly) {
	pushNeuronsStateToDevice(uninitialisedOnly);
}

// ------------------------------------------------------------------------
// OpenCL functions implementation
// ------------------------------------------------------------------------

// Initialize context with the given device
void opencl::setUpContext(cl::Context& context, cl::Device& device, const int deviceIndex) {

	// Getting all platforms to gather devices from
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms); // Gets all the platforms

	assert(platforms.size() > 0);

	// Getting all devices and putting them into a single vector
	std::vector<cl::Device> devices;
	for (int i = 0; i < platforms.size(); i++) {
		std::vector<cl::Device> platformDevices;
		platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
		devices.insert(devices.end(), platformDevices.begin(), platformDevices.end());
	}

	assert(devices.size() > 0);

	// Check if the device exists at the given index
	if (deviceIndex >= devices.size()) {
		assert(deviceIndex >= devices.size());
		device = devices.front();
	}
	else {
		device = devices[deviceIndex]; // We will perform our operations using this device
	}

	context = cl::Context(device);

}

// Create OpenCL program with the specified device
void opencl::createProgram(const char *kernelSource, cl::Program& program, cl::Context& context) {

	// Reading the kernel source for execution
	program = cl::Program(context, kernelSource, true);
	program.build("-cl-std=CL1.2");

}


// Get OpenCL error from error code
std::string opencl::getCLError(cl_int errorCode) {
	switch (errorCode) {
		case CL_SUCCESS:							return "Success!";
		case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
		case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
		case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
		case CL_OUT_OF_RESOURCES:                   return "Out of resources";
		case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
		case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
		case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
		case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
		case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
		case CL_MAP_FAILURE:                        return "Map failure";
		case CL_INVALID_VALUE:                      return "Invalid value";
		case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
		case CL_INVALID_PLATFORM:                   return "Invalid platform";
		case CL_INVALID_DEVICE:                     return "Invalid device";
		case CL_INVALID_CONTEXT:                    return "Invalid context";
		case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
		case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
		case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
		case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:	return "Invalid image format descriptor";
		case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
		case CL_INVALID_SAMPLER:                    return "Invalid sampler";
		case CL_INVALID_BINARY:                     return "Invalid binary";
		case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
		case CL_INVALID_PROGRAM:                    return "Invalid program";
		case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
		case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
		case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
		case CL_INVALID_KERNEL:                     return "Invalid kernel";
		case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
		case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
		case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
		case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
		case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
		case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
		case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
		case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
		case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
		case CL_INVALID_EVENT:                      return "Invalid event";
		case CL_INVALID_OPERATION:                  return "Invalid operation";
		case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
		case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
		case CL_INVALID_MIP_LEVEL:                  return "Invalid mip - map level";
		default:									return "Unknown";
	}
}