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
unsigned int* glbSpkCntE;
cl::Buffer d_glbSpkCntE;
unsigned int* glbSpkE;
cl::Buffer d_glbSpkE;
scalar* VE;
cl::Buffer d_VE;
scalar* RefracTimeE;
cl::Buffer d_RefracTimeE;
unsigned int* glbSpkCntI;
cl::Buffer d_glbSpkCntI;
unsigned int* glbSpkI;
cl::Buffer d_glbSpkI;
scalar* VI;
cl::Buffer d_VI;
scalar* RefracTimeI;
cl::Buffer d_RefracTimeI;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
float* inSynIE;
cl::Buffer d_inSynIE;
float* inSynEE;
cl::Buffer d_inSynEE;
float* inSynII;
cl::Buffer d_inSynII;
float* inSynEI;
cl::Buffer d_inSynEI;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthEE = 4355;
unsigned int* rowLengthEE;
cl::Buffer d_rowLengthEE;
uint32_t* indEE;
cl::Buffer d_indEE;
const unsigned int maxRowLengthEI = 1180;
unsigned int* rowLengthEI;
cl::Buffer d_rowLengthEI;
uint32_t* indEI;
cl::Buffer d_indEI;
const unsigned int maxRowLengthIE = 4341;
unsigned int* rowLengthIE;
cl::Buffer d_rowLengthIE;
uint32_t* indIE;
cl::Buffer d_indIE;
const unsigned int maxRowLengthII = 1172;
unsigned int* rowLengthII;
cl::Buffer d_rowLengthII;
uint32_t* indII;
cl::Buffer d_indII;

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
void pushESpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntE, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntE));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkE, CL_TRUE, 0, 40000 * sizeof(unsigned int), glbSpkE));
    }
}

void pushECurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntE, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntE));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkE, CL_TRUE, 0, glbSpkCntE[0] * sizeof(unsigned int), glbSpkE));
}

void pushVEToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_VE, CL_TRUE, 0, 40000 * sizeof(scalar), VE));
}

void pushCurrentVEToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_VE, CL_TRUE, 0, 40000 * sizeof(scalar), VE));
}

void pushRefracTimeEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_RefracTimeE, CL_TRUE, 0, 40000 * sizeof(scalar), RefracTimeE));
    }
}

void pushCurrentRefracTimeEToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_RefracTimeE, CL_TRUE, 0, 40000 * sizeof(scalar), RefracTimeE));
}

void pushEStateToDevice(bool uninitialisedOnly) {
    pushVEToDevice(uninitialisedOnly);
    pushRefracTimeEToDevice(uninitialisedOnly);
}

void pushISpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntI, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntI));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkI, CL_TRUE, 0, 10000 * sizeof(unsigned int), glbSpkI));
    }
}

void pushICurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntI, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntI));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkI, CL_TRUE, 0, glbSpkCntI[0] * sizeof(unsigned int), glbSpkI));
}

void pushVIToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_VI, CL_TRUE, 0, 10000 * sizeof(scalar), VI));
}

void pushCurrentVIToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_VI, CL_TRUE, 0, 10000 * sizeof(scalar), VI));
}

void pushRefracTimeIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_RefracTimeI, CL_TRUE, 0, 10000 * sizeof(scalar), RefracTimeI));
    }
}

void pushCurrentRefracTimeIToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_RefracTimeI, CL_TRUE, 0, 10000 * sizeof(scalar), RefracTimeI));
}

void pushIStateToDevice(bool uninitialisedOnly) {
    pushVIToDevice(uninitialisedOnly);
    pushRefracTimeIToDevice(uninitialisedOnly);
}

void pushEEConnectivityToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_rowLengthEE, CL_TRUE, 0, 40000 * sizeof(unsigned int), rowLengthEE));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_indEE, CL_TRUE, 0, 174200000 * sizeof(unsigned int), indEE));
}

void pushEIConnectivityToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_rowLengthEI, CL_TRUE, 0, 40000 * sizeof(unsigned int), rowLengthEI));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_indEI, CL_TRUE, 0, 47200000 * sizeof(unsigned int), indEI));
}

void pushIEConnectivityToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_rowLengthIE, CL_TRUE, 0, 10000 * sizeof(unsigned int), rowLengthIE));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_indIE, CL_TRUE, 0, 43410000 * sizeof(unsigned int), indIE));
}

void pushIIConnectivityToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_rowLengthII, CL_TRUE, 0, 10000 * sizeof(unsigned int), rowLengthII));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_indII, CL_TRUE, 0, 11720000 * sizeof(unsigned int), indII));
}

void pushinSynEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynEE, CL_TRUE, 0, 40000 * sizeof(float), inSynEE));
    }
}

void pushEEStateToDevice(bool uninitialisedOnly) {
    pushinSynEEToDevice(uninitialisedOnly);
}

void pushinSynEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynEI, CL_TRUE, 0, 10000 * sizeof(float), inSynEI));
    }
}

void pushEIStateToDevice(bool uninitialisedOnly) {
    pushinSynEIToDevice(uninitialisedOnly);
}

void pushinSynIEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynIE, CL_TRUE, 0, 40000 * sizeof(float), inSynIE));
    }
}

void pushIEStateToDevice(bool uninitialisedOnly) {
    pushinSynIEToDevice(uninitialisedOnly);
}

void pushinSynIIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynII, CL_TRUE, 0, 10000 * sizeof(float), inSynII));
    }
}

void pushIIStateToDevice(bool uninitialisedOnly) {
    pushinSynIIToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullESpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntE, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntE));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkE, CL_TRUE, 0, 40000 * sizeof(unsigned int), glbSpkE));
}

void pullECurrentSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntE, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntE));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkE, CL_TRUE, 0, glbSpkCntE[0] * sizeof(unsigned int), glbSpkE));
}

void pullVEFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_VE, CL_TRUE, 0, 40000 * sizeof(scalar), VE));
}

void pullCurrentVEFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_VE, CL_TRUE, 0, 40000 * sizeof(scalar), VE));
}

void pullRefracTimeEFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_RefracTimeE, CL_TRUE, 0, 40000 * sizeof(scalar), RefracTimeE));
}

void pullCurrentRefracTimeEFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_RefracTimeE, CL_TRUE, 0, 40000 * sizeof(scalar), RefracTimeE));
}

void pullEStateFromDevice() {
    pullVEFromDevice();
    pullRefracTimeEFromDevice();
}

void pullISpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntI, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntI));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkI, CL_TRUE, 0, 10000 * sizeof(unsigned int), glbSpkI));
}

void pullICurrentSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntI, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntI));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkI, CL_TRUE, 0, glbSpkCntI[0] * sizeof(unsigned int), glbSpkI));
}

void pullVIFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_VI, CL_TRUE, 0, 10000 * sizeof(scalar), VI));
}

void pullCurrentVIFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_VI, CL_TRUE, 0, 10000 * sizeof(scalar), VI));
}

void pullRefracTimeIFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_RefracTimeI, CL_TRUE, 0, 10000 * sizeof(scalar), RefracTimeI));
}

void pullCurrentRefracTimeIFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_RefracTimeI, CL_TRUE, 0, 10000 * sizeof(scalar), RefracTimeI));
}

void pullIStateFromDevice() {
    pullVIFromDevice();
    pullRefracTimeIFromDevice();
}

void pullEEConnectivityFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_rowLengthEE, CL_TRUE, 0, 40000 * sizeof(unsigned int), rowLengthEE));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_indEE, CL_TRUE, 0, 174200000 * sizeof(unsigned int), indEE));
}

void pullEIConnectivityFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_rowLengthEI, CL_TRUE, 0, 40000 * sizeof(unsigned int), rowLengthEI));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_indEI, CL_TRUE, 0, 47200000 * sizeof(unsigned int), indEI));
}

void pullIEConnectivityFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_rowLengthIE, CL_TRUE, 0, 10000 * sizeof(unsigned int), rowLengthIE));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_indIE, CL_TRUE, 0, 43410000 * sizeof(unsigned int), indIE));
}

void pullIIConnectivityFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_rowLengthII, CL_TRUE, 0, 10000 * sizeof(unsigned int), rowLengthII));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_indII, CL_TRUE, 0, 11720000 * sizeof(unsigned int), indII));
}

void pullinSynEEFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynEE, CL_TRUE, 0, 40000 * sizeof(float), inSynEE));
}

void pullEEStateFromDevice() {
    pullinSynEEFromDevice();
}

void pullinSynEIFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynEI, CL_TRUE, 0, 10000 * sizeof(float), inSynEI));
}

void pullEIStateFromDevice() {
    pullinSynEIFromDevice();
}

void pullinSynIEFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynIE, CL_TRUE, 0, 40000 * sizeof(float), inSynIE));
}

void pullIEStateFromDevice() {
    pullinSynIEFromDevice();
}

void pullinSynIIFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynII, CL_TRUE, 0, 10000 * sizeof(float), inSynII));
}

void pullIIStateFromDevice() {
    pullinSynIIFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getECurrentSpikes() {
    return  glbSpkE;
}

unsigned int& getECurrentSpikeCount() {
    return glbSpkCntE[0];
}

scalar* getCurrentVE() {
    return VE;
}

scalar* getCurrentRefracTimeE() {
    return RefracTimeE;
}

unsigned int* getICurrentSpikes() {
    return  glbSpkI;
}

unsigned int& getICurrentSpikeCount() {
    return glbSpkCntI[0];
}

scalar* getCurrentVI() {
    return VI;
}

scalar* getCurrentRefracTimeI() {
    return RefracTimeI;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushEStateToDevice(uninitialisedOnly);
    pushIStateToDevice(uninitialisedOnly);
    pushEEStateToDevice(uninitialisedOnly);
    pushEIStateToDevice(uninitialisedOnly);
    pushIEStateToDevice(uninitialisedOnly);
    pushIIStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushEEConnectivityToDevice(uninitialisedOnly);
    pushEIConnectivityToDevice(uninitialisedOnly);
    pushIEConnectivityToDevice(uninitialisedOnly);
    pushIIConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pullEStateFromDevice();
    pullIStateFromDevice();
    pullEEStateFromDevice();
    pullEIStateFromDevice();
    pullIEStateFromDevice();
    pullIIStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullECurrentSpikesFromDevice();
    pullICurrentSpikesFromDevice();
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
    glbSpkCntE = (unsigned int*)malloc(1 * sizeof(unsigned int));
    d_glbSpkCntE = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof (unsigned int), glbSpkCntE);
    glbSpkE = (unsigned int*)malloc(40000 * sizeof(unsigned int));
    d_glbSpkE = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 40000 * sizeof (unsigned int), glbSpkE);
    VE = (scalar*)malloc(40000 * sizeof(scalar));
    d_VE = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 40000 * sizeof (scalar), VE);
    RefracTimeE = (scalar*)malloc(40000 * sizeof(scalar));
    d_RefracTimeE = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 40000 * sizeof (scalar), RefracTimeE);
    glbSpkCntI = (unsigned int*)malloc(1 * sizeof(unsigned int));
    d_glbSpkCntI = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof (unsigned int), glbSpkCntI);
    glbSpkI = (unsigned int*)malloc(10000 * sizeof(unsigned int));
    d_glbSpkI = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof (unsigned int), glbSpkI);
    VI = (scalar*)malloc(10000 * sizeof(scalar));
    d_VI = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof (scalar), VI);
    RefracTimeI = (scalar*)malloc(10000 * sizeof(scalar));
    d_RefracTimeI = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof (scalar), RefracTimeI);
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    inSynIE = (float*)malloc(40000 * sizeof(float));
    d_inSynIE = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 40000 * sizeof (float), inSynIE);
    inSynEE = (float*)malloc(40000 * sizeof(float));
    d_inSynEE = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 40000 * sizeof (float), inSynEE);
    inSynII = (float*)malloc(10000 * sizeof(float));
    d_inSynII = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof (float), inSynII);
    inSynEI = (float*)malloc(10000 * sizeof(float));
    d_inSynEI = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof (float), inSynEI);
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    rowLengthEE = (unsigned int*)malloc(40000 * sizeof(unsigned int));
    d_rowLengthEE = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 40000 * sizeof (unsigned int), rowLengthEE);
    indEE = (uint32_t*)malloc(174200000 * sizeof(uint32_t));
    d_indEE = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 174200000 * sizeof (uint32_t), indEE);
    rowLengthEI = (unsigned int*)malloc(40000 * sizeof(unsigned int));
    d_rowLengthEI = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 40000 * sizeof (unsigned int), rowLengthEI);
    indEI = (uint32_t*)malloc(47200000 * sizeof(uint32_t));
    d_indEI = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 47200000 * sizeof (uint32_t), indEI);
    rowLengthIE = (unsigned int*)malloc(10000 * sizeof(unsigned int));
    d_rowLengthIE = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof (unsigned int), rowLengthIE);
    indIE = (uint32_t*)malloc(43410000 * sizeof(uint32_t));
    d_indIE = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 43410000 * sizeof (uint32_t), indIE);
    rowLengthII = (unsigned int*)malloc(10000 * sizeof(unsigned int));
    d_rowLengthII = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof (unsigned int), rowLengthII);
    indII = (uint32_t*)malloc(11720000 * sizeof(uint32_t));
    d_indII = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 11720000 * sizeof (uint32_t), indII);
    
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
    free(glbSpkCntE);
    free(glbSpkE);
    free(VE);
    free(RefracTimeE);
    free(glbSpkCntI);
    free(glbSpkI);
    free(VI);
    free(RefracTimeI);
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    free(inSynIE);
    free(inSynEE);
    free(inSynII);
    free(inSynEI);
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    free(rowLengthEE);
    free(indEE);
    free(rowLengthEI);
    free(indEI);
    free(rowLengthIE);
    free(indIE);
    free(rowLengthII);
    free(indII);
    
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

