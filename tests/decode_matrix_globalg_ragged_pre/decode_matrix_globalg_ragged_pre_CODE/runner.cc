#include "definitionsInternal.h"

extern "C" {
    // OpenCL variables
    cl::Context clContext;
    cl::Device clDevice;
    cl::CommandQueue commandQueue;
    
    // OpenCL programs
    cl::Program initProgram;
    cl::Program updateNeuronsProgram;
    cl::Program updateSynapsesProgram;
    
    // OpenCL kernels
    cl::Kernel updateNeuronsKernel;
    cl::Kernel updatePresynapticKernel;
    cl::Kernel updatePostsynapticKernel;
    cl::Kernel updateSynapseDynamicsKernel;
    cl::Kernel initializeKernel;
    cl::Kernel initializeSparseKernel;
    cl::Kernel preNeuronResetKernel;
    cl::Kernel preSynapseResetKernel;
}

// Initializing OpenCL programs so that they can be used to run the kernels
void initPrograms() {
    opencl::setUpContext(clContext, clDevice, DEVICE_INDEX);
    commandQueue = cl::CommandQueue(clContext, clDevice);
    
    // Create programs for kernels
    opencl::createProgram(initProgramSrc, initProgram, clContext);
    opencl::createProgram(updateNeuronsProgramSrc, updateNeuronsProgram, clContext);
    opencl::createProgram(updateSynapsesProgramSrc, updateSynapsesProgram, clContext);
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
    for (const auto& platform : platforms) {
        std::vector<cl::Device> platformDevices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
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
void opencl::createProgram(const char* kernelSource, cl::Program& program, cl::Context& context) {
    // Reading the kernel source for execution
    program = cl::Program(context, kernelSource, true);
    program.build("-cl-std=CL1.2");
}

// Get OpenCL error as string
const char* opencl::clGetErrorString(cl_int error) {
    switch(error) {
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -64: return "CL_INVALID_PROPERTY";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -59: return "CL_INVALID_OPERATION";
        case -58: return "CL_INVALID_EVENT";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -48: return "CL_INVALID_KERNEL";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -44: return "CL_INVALID_PROGRAM";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -42: return "CL_INVALID_BINARY";
        case -41: return "CL_INVALID_SAMPLER";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -37: return "CL_INVALID_HOST_PTR";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -34: return "CL_INVALID_CONTEXT";
        case -33: return "CL_INVALID_DEVICE";
        case -32: return "CL_INVALID_PLATFORM";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -30: return "CL_INVALID_VALUE";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -12: return "CL_MAP_FAILURE";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case 0: return "CL_SUCCESS";
        default: return "Unknown OpenCL error";
    }
}

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
unsigned long long iT;
float t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double neuronUpdateTime;
double initTime;
double presynapticUpdateTime;
double postsynapticUpdateTime;
double synapseDynamicsTime;
double initSparseTime;
// ------------------------------------------------------------------------
// remote neuron groups
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntPost;
cl::Buffer d_glbSpkCntPost;
unsigned int* glbSpkPost;
cl::Buffer d_glbSpkPost;
scalar* xPost;
cl::Buffer d_xPost;
unsigned int* glbSpkCntPre;
cl::Buffer d_glbSpkCntPre;
unsigned int* glbSpkPre;
cl::Buffer d_glbSpkPre;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
float* inSynSyn;
cl::Buffer d_inSynSyn;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthSyn = 4;
unsigned int* rowLengthSyn;
cl::Buffer d_rowLengthSyn;
uint32_t* indSyn;
cl::Buffer d_indSyn;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushPostSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntPost, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntPost));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkPost, CL_TRUE, 0, 4 * sizeof(unsigned int), glbSpkPost));
    }
}

void pushPostCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntPost, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntPost));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntPost, CL_TRUE, 0, glbSpkCntPost[0] * sizeof(unsigned int), glbSpkPost));
}

void pushxPostToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_xPost, CL_TRUE, 0, 4 * sizeof(scalar), xPost));
    }
}

void pushCurrentxPostToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_xPost, CL_TRUE, 0, 4 * sizeof(scalar), xPost));
}

void pushPostStateToDevice(bool uninitialisedOnly) {
    pushxPostToDevice(uninitialisedOnly);
}

void pushPreSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntPre, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntPre));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkPre, CL_TRUE, 0, 10 * sizeof(unsigned int), glbSpkPre));
    }
}

void pushPreCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntPre, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntPre));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntPre, CL_TRUE, 0, glbSpkCntPre[0] * sizeof(unsigned int), glbSpkPre));
}

void pushPreStateToDevice(bool uninitialisedOnly) {
}

void pushSynConnectivityToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_rowLengthSyn, CL_TRUE, 0, 10 * sizeof(unsigned int), rowLengthSyn));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_indSyn, CL_TRUE, 0, 40 * sizeof(unsigned int), indSyn));
}

void pushinSynSynToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynSyn, CL_TRUE, 0, 4 * sizeof(float), inSynSyn));
    }
}

void pushSynStateToDevice(bool uninitialisedOnly) {
    pushinSynSynToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullPostSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntPost, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntPost));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkPost, CL_TRUE, 0, 4 * sizeof(unsigned int), glbSpkPost));
}

void pullPostCurrentSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntPost, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntPost));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntPost, CL_TRUE, 0, glbSpkCntPost[0] * sizeof(unsigned int), glbSpkPost));
}

void pullxPostFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_xPost, CL_TRUE, 0, 4 * sizeof(scalar), xPost));
}

void pullCurrentxPostFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_xPost, CL_TRUE, 0, 4 * sizeof(scalar), xPost));
}

void pullPostStateFromDevice() {
    pullxPostFromDevice();
}

void pullPreSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntPre, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntPre));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkPre, CL_TRUE, 0, 10 * sizeof(unsigned int), glbSpkPre));
}

void pullPreCurrentSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntPre, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntPre));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntPre, CL_TRUE, 0, glbSpkCntPre[0] * sizeof(unsigned int), glbSpkPre));
}

void pullPreStateFromDevice() {
}

void pullSynConnectivityFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_rowLengthSyn, CL_TRUE, 0, 10 * sizeof(unsigned int), rowLengthSyn));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_indSyn, CL_TRUE, 0, 40 * sizeof(unsigned int), indSyn));
}

void pullinSynSynFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynSyn, CL_TRUE, 0, 4 * sizeof(float), inSynSyn));
}

void pullSynStateFromDevice() {
    pullinSynSynFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getPostCurrentSpikes() {
    return  glbSpkPost;
}

unsigned int& getPostCurrentSpikeCount() {
    return glbSpkCntPost[0];
}

scalar* getCurrentxPost() {
    return xPost;
}

unsigned int* getPreCurrentSpikes() {
    return  glbSpkPre;
}

unsigned int& getPreCurrentSpikeCount() {
    return glbSpkCntPre[0];
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushPostStateToDevice(uninitialisedOnly);
    pushPreStateToDevice(uninitialisedOnly);
    pushSynStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushSynConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pullPostStateFromDevice();
    pullPreStateFromDevice();
    pullSynStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullPostCurrentSpikesFromDevice();
    pullPreCurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
}

void allocateMem() {
    initPrograms();
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // remote neuron groups
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    glbSpkCntPost = (unsigned int*)malloc(1 * sizeof(unsigned int));
    d_glbSpkCntPost = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof (unsigned int), glbSpkCntPost);
    glbSpkPost = (unsigned int*)malloc(4 * sizeof(unsigned int));
    d_glbSpkPost = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4 * sizeof (unsigned int), glbSpkPost);
    xPost = (scalar*)malloc(4 * sizeof(scalar));
    d_xPost = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4 * sizeof (scalar), xPost);
    glbSpkCntPre = (unsigned int*)malloc(1 * sizeof(unsigned int));
    d_glbSpkCntPre = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof (unsigned int), glbSpkCntPre);
    glbSpkPre = (unsigned int*)malloc(10 * sizeof(unsigned int));
    d_glbSpkPre = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (unsigned int), glbSpkPre);
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    inSynSyn = (float*)malloc(4 * sizeof(float));
    d_inSynSyn = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4 * sizeof (float), inSynSyn);
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    rowLengthSyn = (unsigned int*)malloc(10 * sizeof(unsigned int));
    d_rowLengthSyn = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (unsigned int), rowLengthSyn);
    indSyn = (uint32_t*)malloc(40 * sizeof(uint32_t));
    d_indSyn = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 40 * sizeof (uint32_t), indSyn);
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // OpenCL kernels initialization
    // ------------------------------------------------------------------------
    initProgramKernels();
    updateNeuronsProgramKernels();
    updateSynapsesProgramKernels();
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // remote neuron groups
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    free(glbSpkCntPost);
    free(glbSpkPost);
    free(xPost);
    free(glbSpkCntPre);
    free(glbSpkPre);
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    free(inSynSyn);
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    free(rowLengthSyn);
    free(indSyn);
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t);
    iT++;
    t = iT*DT;
}

