#include "definitionsInternal.h"

extern "C" {
	unsigned int* dd_glbSpkCntNeurons;
	unsigned int* dd_glbSpkNeurons;
	scalar* dd_VNeurons;
	scalar* dd_UNeurons;
	scalar* dd_aNeurons;
	scalar* dd_bNeurons;
	scalar* dd_cNeurons;
	scalar* dd_dNeurons;
	unsigned long long iT;
	float t;

	// Buffers
	// initializeKernel buffers
	cl::Buffer b_init_glbSpkCntNeurons;
	cl::Buffer b_init_glbSpkNeurons;
	cl::Buffer b_init_VNeurons;
	cl::Buffer b_init_UNeurons;
	// updateNeuronsKernels buffers
	cl::Buffer b_glbSpkCntNeurons;
	cl::Buffer b_glbSpkNeurons;
	cl::Buffer b_VNeurons;
	cl::Buffer b_UNeurons;
	cl::Buffer b_aNeurons;
	cl::Buffer b_bNeurons;
	cl::Buffer b_cNeurons;
	cl::Buffer b_dNeurons;

	// OpenCL variables
	cl::Program initProgram;
	cl::Context initContext;
	cl::Device initDevice;

	cl::Program unProgram;
	cl::Context unContext;
	cl::Device unDevice;
}

// Allocating memory to pointers
void allocateMem() {

	// Allocating memory to host pointers
	dd_glbSpkCntNeurons = (unsigned int*)malloc(1 * sizeof(unsigned int));
	dd_glbSpkNeurons = (unsigned int*)malloc(NSIZE * sizeof(unsigned int));
	dd_VNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	dd_UNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	dd_aNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	dd_bNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	dd_cNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	dd_dNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));

	// Buffers
	// initializeKernel buffers
	b_init_glbSpkCntNeurons = cl::Buffer(initContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 1 * sizeof(dd_glbSpkCntNeurons), dd_glbSpkCntNeurons);
	b_init_glbSpkNeurons = cl::Buffer(initContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_glbSpkNeurons), dd_glbSpkNeurons);
	b_init_VNeurons = cl::Buffer(initContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(scalar), dd_VNeurons);
	b_init_UNeurons = cl::Buffer(initContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(scalar), dd_UNeurons);
	// updateNeuronsKernels buffers
	b_glbSpkCntNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 1 * sizeof(dd_glbSpkCntNeurons), dd_glbSpkCntNeurons);
	b_glbSpkNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_glbSpkNeurons), dd_glbSpkNeurons);
	b_VNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_VNeurons), dd_VNeurons);
	b_UNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_UNeurons), dd_UNeurons);
	b_aNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_aNeurons), dd_aNeurons);
	b_bNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_bNeurons), dd_bNeurons);
	b_cNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_cNeurons), dd_cNeurons);
	b_dNeurons = cl::Buffer(unContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NSIZE * sizeof(dd_dNeurons), dd_dNeurons);
}

// Initializing kernel programs so that they can be used to run the kernels
void initKernelPrograms() {
	opencl::createProgram("init.cl", initProgram, initContext, initDevice, 1);
	opencl::createProgram("updateNeuronsKernels.cl", unProgram, unContext, unDevice, 1);
}

void stepTime() {
	updateNeurons(t);
	iT++;
	t = iT * DT;
}

scalar* getCurrentVNeurons() {
	return dd_VNeurons;
}

void pullCurrentVNeuronsFromDevice() {

}


/*
OpenCL function implementations
*/

// Create OpenCL program with the specified device
void opencl::createProgram(const std::string& filename, cl::Program& program, cl::Context& context, cl::Device& device, const int deviceIndex) {

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

	// Reading the kernel file for execution
	std::ifstream kernelFile(filename);
	std::string srcString(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));


	cl::Program::Sources programSources(1, std::make_pair(srcString.c_str(), srcString.length() + 1));

	context = cl::Context(device);
	program = cl::Program(context, programSources);

	program.build("-cl-std=CL1.2");

}


// Get OpenCL error from error code
std::string opencl::getCLError(cl_int errorCode) {
	switch (errorCode) {
	case CL_SUCCESS:                            return "Success!";
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
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
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
	default: return "Unknown";
	}
}