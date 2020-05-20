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
} // extern "C"

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
unsigned int* glbSpkCntpost;
cl::Buffer d_glbSpkCntpost;
unsigned int* glbSpkpost;
cl::Buffer d_glbSpkpost;
unsigned int spkQuePtrpost;
cl::Buffer d_spkQuePtrpost;
unsigned int* glbSpkCntpre;
cl::Buffer d_glbSpkCntpre;
unsigned int* glbSpkpre;
cl::Buffer d_glbSpkpre;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
float* inSynsyn;
cl::Buffer d_inSynsyn;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthsyn = 1;
unsigned int* rowLengthsyn;
cl::Buffer d_rowLengthsyn;
uint32_t* indsyn;
cl::Buffer d_indsyn;
unsigned int* colLengthsyn;
cl::Buffer d_colLengthsyn;
unsigned int* remapsyn;
cl::Buffer d_remapsyn;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
scalar* wsyn;
cl::Buffer d_wsyn;
scalar* ssyn;
cl::Buffer d_ssyn;

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushpostSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntpost, CL_TRUE, 0, 21 * sizeof(unsigned int), glbSpkCntpost));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkpost, CL_TRUE, 0, 210 * sizeof(unsigned int), glbSpkpost));
    }
}

void pushpostCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntpost, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntpost));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkpost, CL_TRUE, 0, 10 * sizeof(unsigned int), glbSpkpost));
}

void pushpostStateToDevice(bool uninitialisedOnly) {
}

void pushpreSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntpre, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntpre));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkpre, CL_TRUE, 0, 10 * sizeof(unsigned int), glbSpkpre));
    }
}

void pushpreCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntpre, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntpre));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkpre, CL_TRUE, 0, glbSpkCntpre[0] * sizeof(unsigned int), glbSpkpre));
}

void pushpreStateToDevice(bool uninitialisedOnly) {
}

void pushsynConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_rowLengthsyn, CL_TRUE, 0, 10 * sizeof(unsigned int), rowLengthsyn));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_indsyn, CL_TRUE, 0, 10 * sizeof(unsigned int), indsyn));
    }
}

void pushwsynToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_wsyn, CL_TRUE, 0, 10 * sizeof(scalar), wsyn));
    }
}

void pushssynToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_ssyn, CL_TRUE, 0, 210 * sizeof(scalar), ssyn));
    }
}

void pushinSynsynToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynsyn, CL_TRUE, 0, 10 * sizeof(float), inSynsyn));
    }
}

void pushsynStateToDevice(bool uninitialisedOnly) {
    pushwsynToDevice(uninitialisedOnly);
    pushssynToDevice(uninitialisedOnly);
    pushinSynsynToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullpostSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntpost, CL_TRUE, 0, 21 * sizeof(unsigned int), glbSpkCntpost));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkpost, CL_TRUE, 0, 210 * sizeof(unsigned int), glbSpkpost));
}

void pullpostCurrentSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntpost, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntpost));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkpost, CL_TRUE, 0, 10 * sizeof(unsigned int), glbSpkpost));
}

void pullpostStateFromDevice() {
}

void pullpreSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntpre, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntpre));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkpre, CL_TRUE, 0, 10 * sizeof(unsigned int), glbSpkpre));
}

void pullpreCurrentSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntpre, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntpre));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkpre, CL_TRUE, 0, glbSpkCntpre[0] * sizeof(unsigned int), glbSpkpre));
}

void pullpreStateFromDevice() {
}

void pullsynConnectivityFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_rowLengthsyn, CL_TRUE, 0, 10 * sizeof(unsigned int), rowLengthsyn));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_indsyn, CL_TRUE, 0, 10 * sizeof(unsigned int), indsyn));
}

void pullwsynFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_wsyn, CL_TRUE, 0, 10 * sizeof(scalar), wsyn));
}

void pullssynFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_ssyn, CL_TRUE, 0, 210 * sizeof(scalar), ssyn));
}

void pullinSynsynFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynsyn, CL_TRUE, 0, 10 * sizeof(float), inSynsyn));
}

void pullsynStateFromDevice() {
    pullwsynFromDevice();
    pullssynFromDevice();
    pullinSynsynFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getpostCurrentSpikes() {
    return  (glbSpkpost + (spkQuePtrpost * 10));
}

unsigned int& getpostCurrentSpikeCount() {
    return glbSpkCntpost[spkQuePtrpost];
}

unsigned int* getpreCurrentSpikes() {
    return  glbSpkpre;
}

unsigned int& getpreCurrentSpikeCount() {
    return glbSpkCntpre[0];
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushpostStateToDevice(uninitialisedOnly);
    pushpreStateToDevice(uninitialisedOnly);
    pushsynStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushsynConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pullpostStateFromDevice();
    pullpreStateFromDevice();
    pullsynStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullpostCurrentSpikesFromDevice();
    pullpreCurrentSpikesFromDevice();
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
    glbSpkCntpost = (unsigned int*)calloc(21, sizeof(unsigned int));
    d_glbSpkCntpost = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 21 * sizeof (unsigned int), glbSpkCntpost);
    glbSpkpost = (unsigned int*)calloc(210, sizeof(unsigned int));
    d_glbSpkpost = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 210 * sizeof (unsigned int), glbSpkpost);
    glbSpkCntpre = (unsigned int*)calloc(1, sizeof(unsigned int));
    d_glbSpkCntpre = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof (unsigned int), glbSpkCntpre);
    glbSpkpre = (unsigned int*)calloc(10, sizeof(unsigned int));
    d_glbSpkpre = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (unsigned int), glbSpkpre);
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    inSynsyn = (float*)calloc(10, sizeof(float));
    d_inSynsyn = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (float), inSynsyn);
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    rowLengthsyn = (unsigned int*)calloc(10, sizeof(unsigned int));
    d_rowLengthsyn = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (unsigned int), rowLengthsyn);
    indsyn = (uint32_t*)calloc(10, sizeof(uint32_t));
    d_indsyn = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (uint32_t), indsyn);
    colLengthsyn = (unsigned int*)calloc(10, sizeof(unsigned int));
    d_colLengthsyn = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (unsigned int), colLengthsyn);
    remapsyn = (unsigned int*)calloc(10, sizeof(unsigned int));
    d_remapsyn = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (unsigned int), remapsyn);
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    wsyn = (scalar*)calloc(10, sizeof(scalar));
    d_wsyn = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (scalar), wsyn);
    ssyn = (scalar*)calloc(210, sizeof(scalar));
    d_ssyn = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 210 * sizeof (scalar), ssyn);
    
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
    free(glbSpkCntpost);
    free(glbSpkpost);
    free(glbSpkCntpre);
    free(glbSpkpre);
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    free(inSynsyn);
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    free(rowLengthsyn);
    free(indsyn);
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    free(wsyn);
    free(ssyn);
    
}

void stepTime() {
    updateSynapses(t);
    spkQuePtrpost = (spkQuePtrpost + 1) % 21;
    updateNeurons(t);
    iT++;
    t = iT*DT;
}

