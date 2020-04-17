#include "definitionsInternal.h"
#include <iostream>

extern "C" {
    unsigned long long iT;
    float t;

    // Buffers and host variables
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    unsigned int* glbSpkCntExc;
    cl::Buffer d_glbSpkCntExc;
    unsigned int* glbSpkExc;
    cl::Buffer d_glbSpkExc;
    scalar* VExc;
    cl::Buffer d_VExc;
    scalar* UExc;
    cl::Buffer d_UExc;
    // current source variables
    unsigned int* glbSpkCntInh;
    cl::Buffer d_glbSpkCntInh;
    unsigned int* glbSpkInh;
    cl::Buffer d_glbSpkInh;
    scalar* VInh;
    cl::Buffer d_VInh;
    scalar* UInh;
    cl::Buffer d_UInh;
    // current source variables

    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    float* inSynInh_Exc;
    cl::Buffer d_inSynInh_Exc;
    float* inSynExc_Exc;
    cl::Buffer d_inSynExc_Exc;
    float* inSynInh_Inh;
    cl::Buffer d_inSynInh_Inh;
    float* inSynExc_Inh;
    cl::Buffer d_inSynExc_Inh;

    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    const unsigned int maxRowLengthExc_Exc = 953;
    unsigned int* rowLengthExc_Exc;
    cl::Buffer d_rowLengthExc_Exc;
    uint32_t* indExc_Exc;
    cl::Buffer d_indExc_Exc;
    const unsigned int maxRowLengthExc_Inh = 279;
    unsigned int* rowLengthExc_Inh;
    cl::Buffer d_rowLengthExc_Inh;
    uint32_t* indExc_Inh;
    cl::Buffer d_indExc_Inh;
    const unsigned int maxRowLengthInh_Exc = 946;
    unsigned int* rowLengthInh_Exc;
    cl::Buffer d_rowLengthInh_Exc;
    uint32_t* indInh_Exc;
    cl::Buffer d_indInh_Exc;
    const unsigned int maxRowLengthInh_Inh = 275;
    unsigned int* rowLengthInh_Inh;
    cl::Buffer d_rowLengthInh_Inh;
    uint32_t* indInh_Inh;
    cl::Buffer d_indInh_Inh;
}

// Initializing kernel programs so that they can be used to run the kernels
void initPrograms() {
    opencl::setUpContext(clContext, clDevice, DEVICE_INDEX);
    // Create programs for kernels
    opencl::createProgram(initKernelSource, initProgram, clContext);
    opencl::createProgram(updateNeuronsKernelSource, unProgram, clContext);
    commandQueue = cl::CommandQueue(clContext, clDevice);
}

// Allocating memory to pointers
void allocateMem() {
    initPrograms();

    d_rng = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof(clrngPhilox432Stream), rng);
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    glbSpkCntExc = (unsigned int*)malloc(1 * sizeof(unsigned int));
    d_glbSpkCntExc = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof(unsigned int), glbSpkCntExc);
    glbSpkExc = (unsigned int*)malloc(1 * sizeof(unsigned int));
    d_glbSpkExc = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof(unsigned int), glbSpkExc);
    VExc = (scalar*)malloc(8000 * sizeof(scalar));
    d_VExc = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 8000 * sizeof(scalar), VExc);
    UExc = (scalar*)malloc(8000 * sizeof(scalar));
    d_UExc = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 8000 * sizeof(scalar), UExc);
    // current source variables
    glbSpkCntInh = (unsigned int*)malloc(1 * sizeof(unsigned int));
    d_glbSpkCntInh = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof(unsigned int), glbSpkCntInh);
    glbSpkInh = (unsigned int*)malloc(2000 * sizeof(unsigned int));
    d_glbSpkInh = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 2000 * sizeof(unsigned int), glbSpkInh);
    VInh = (scalar*)malloc(2000 * sizeof(scalar));
    d_VInh = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 2000 * sizeof(scalar), VInh);
    UInh = (scalar*)malloc(2000 * sizeof(scalar));
    d_UInh = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 2000 * sizeof(scalar), UInh);
    // current source variables

    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    inSynInh_Exc = (float*)malloc(8000 * sizeof(float));
    d_inSynInh_Exc = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 8000 * sizeof(float), inSynInh_Exc);
    inSynExc_Exc = (float*)malloc(8000 * sizeof(float));
    d_inSynExc_Exc = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 8000 * sizeof(float), inSynExc_Exc);
    inSynInh_Inh = (float*)malloc(2000 * sizeof(float));
    d_inSynInh_Inh = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 2000 * sizeof(float), inSynInh_Inh);
    inSynExc_Inh = (float*)malloc(2000 * sizeof(float));
    d_inSynExc_Inh = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 2000 * sizeof(float), inSynExc_Inh);

    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    rowLengthExc_Exc = (unsigned int*)malloc(8000 * sizeof(unsigned int));
    d_rowLengthExc_Exc = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 8000 * sizeof(unsigned int), rowLengthExc_Exc);
    indExc_Exc = (uint32_t*)malloc(7624000 * sizeof(uint32_t));
    d_indExc_Exc = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 7624000 * sizeof(uint32_t), indExc_Exc);
    rowLengthExc_Inh = (unsigned int*)malloc(8000 * sizeof(unsigned int));
    d_rowLengthExc_Inh = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 8000 * sizeof(unsigned int), rowLengthExc_Inh);
    indExc_Inh = (uint32_t*)malloc(2232000 * sizeof(uint32_t));
    d_indExc_Inh = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 2232000 * sizeof(uint32_t), indExc_Inh);
    rowLengthInh_Exc = (unsigned int*)malloc(2000 * sizeof(unsigned int));
    d_rowLengthInh_Exc = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 2000 * sizeof(unsigned int), rowLengthInh_Exc);
    indInh_Exc = (uint32_t*)malloc(1892000 * sizeof(uint32_t));
    d_indInh_Exc = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1892000 * sizeof(uint32_t), indInh_Exc);
    rowLengthInh_Inh = (unsigned int*)malloc(2000 * sizeof(unsigned int));
    d_rowLengthInh_Inh = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 2000 * sizeof(unsigned int), rowLengthInh_Inh);
    indInh_Inh = (uint32_t*)malloc(550000 * sizeof(uint32_t));
    d_indInh_Inh = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 550000 * sizeof(uint32_t), indInh_Inh);

    // Initializing kernels
    initInitKernel();
    initUpdateNeuronsKernels();
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t);
    iT++;
    t = iT * DT;
}

// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getExcCurrentSpikes() {
    return glbSpkExc;
}
unsigned int* getInhCurrentSpikes() {
    return glbSpkInh;
}

void pullExcCurrentSpikesFromDevice() {
    commandQueue.enqueueReadBuffer(d_glbSpkCntExc, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntExc);
}
void pullInhCurrentSpikesFromDevice() {
    commandQueue.enqueueReadBuffer(d_glbSpkCntInh, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntInh);
}

unsigned int& getInhCurrentSpikeCount() {
    return glbSpkCntInh[0];
}
unsigned int& getExcCurrentSpikeCount() {
    return glbSpkCntExc[0];
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

    std::cout << "DEVICE: " << device.getInfo<CL_DEVICE_NAME>();

    context = cl::Context(device);

}

// Create OpenCL program with the specified device
void opencl::createProgram(const char* kernelSource, cl::Program& program, cl::Context& context) {

    // Reading the kernel source for execution
    program = cl::Program(context, kernelSource, true);
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
        default:                                    return "Unknown";
    }
}