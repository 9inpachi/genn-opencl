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
void opencl::createProgram(const char* kernelSource, cl::Program& program, cl::Context& context) {
    // Reading the kernel source for execution
    program = cl::Program(context, kernelSource, true);
    program.build("-cl-std=CL1.2");
}

// Get OpenCL error as string
char* opencl::clGetErrorString(cl_int error) {
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
unsigned int* glbSpkCntInput;
cl::Buffer d_glbSpkCntInput;
unsigned int* glbSpkInput;
cl::Buffer d_glbSpkInput;
unsigned int spkQuePtrInput;
cl::Buffer d_spkQuePtrInput;
scalar* VInput;
cl::Buffer d_VInput;
scalar* UInput;
cl::Buffer d_UInput;
// current source variables
unsigned int* glbSpkCntInter;
cl::Buffer d_glbSpkCntInter;
unsigned int* glbSpkInter;
cl::Buffer d_glbSpkInter;
scalar* VInter;
cl::Buffer d_VInter;
scalar* UInter;
cl::Buffer d_UInter;
unsigned int* glbSpkCntOutput;
cl::Buffer d_glbSpkCntOutput;
unsigned int* glbSpkOutput;
cl::Buffer d_glbSpkOutput;
scalar* VOutput;
cl::Buffer d_VOutput;
scalar* UOutput;
cl::Buffer d_UOutput;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
float* inSynInputInter;
cl::Buffer d_inSynInputInter;
float* inSynInterOutput;
cl::Buffer d_inSynInterOutput;
float* inSynInputOutput;
cl::Buffer d_inSynInputOutput;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

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
void pushInputSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntInput, CL_TRUE, 0, 7 * sizeof(unsigned int), glbSpkCntInput));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkInput, CL_TRUE, 0, 3500 * sizeof(unsigned int), glbSpkInput));
    }
}

void pushInputCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushVInputToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_VInput, CL_TRUE, 0, 500 * sizeof(scalar), VInput));
    }
}

void pushCurrentVInputToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_VInput, CL_TRUE, 0, 500 * sizeof(scalar), VInput));
}

void pushUInputToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_UInput, CL_TRUE, 0, 500 * sizeof(scalar), UInput));
    }
}

void pushCurrentUInputToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_UInput, CL_TRUE, 0, 500 * sizeof(scalar), UInput));
}

void pushInputStateToDevice(bool uninitialisedOnly) {
    pushVInputToDevice(uninitialisedOnly);
    pushUInputToDevice(uninitialisedOnly);
}

void pushInputCurrentSourceStateToDevice(bool uninitialisedOnly) {
}

void pushInterSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntInter, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntInter));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkInter, CL_TRUE, 0, 500 * sizeof(unsigned int), glbSpkInter));
    }
}

void pushInterCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushVInterToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_VInter, CL_TRUE, 0, 500 * sizeof(scalar), VInter));
    }
}

void pushCurrentVInterToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_VInter, CL_TRUE, 0, 500 * sizeof(scalar), VInter));
}

void pushUInterToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_UInter, CL_TRUE, 0, 500 * sizeof(scalar), UInter));
    }
}

void pushCurrentUInterToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_UInter, CL_TRUE, 0, 500 * sizeof(scalar), UInter));
}

void pushInterStateToDevice(bool uninitialisedOnly) {
    pushVInterToDevice(uninitialisedOnly);
    pushUInterToDevice(uninitialisedOnly);
}

void pushOutputSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntOutput, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntOutput));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkOutput, CL_TRUE, 0, 500 * sizeof(unsigned int), glbSpkOutput));
    }
}

void pushOutputCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushVOutputToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_VOutput, CL_TRUE, 0, 500 * sizeof(scalar), VOutput));
    }
}

void pushCurrentVOutputToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_VOutput, CL_TRUE, 0, 500 * sizeof(scalar), VOutput));
}

void pushUOutputToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_UOutput, CL_TRUE, 0, 500 * sizeof(scalar), UOutput));
    }
}

void pushCurrentUOutputToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_UOutput, CL_TRUE, 0, 500 * sizeof(scalar), UOutput));
}

void pushOutputStateToDevice(bool uninitialisedOnly) {
    pushVOutputToDevice(uninitialisedOnly);
    pushUOutputToDevice(uninitialisedOnly);
}

void pushinSynInputInterToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynInputInter, CL_TRUE, 0, 500 * sizeof(float), inSynInputInter));
    }
}

void pushInputInterStateToDevice(bool uninitialisedOnly) {
    pushinSynInputInterToDevice(uninitialisedOnly);
}

void pushinSynInputOutputToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynInputOutput, CL_TRUE, 0, 500 * sizeof(float), inSynInputOutput));
    }
}

void pushInputOutputStateToDevice(bool uninitialisedOnly) {
    pushinSynInputOutputToDevice(uninitialisedOnly);
}

void pushinSynInterOutputToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynInterOutput, CL_TRUE, 0, 500 * sizeof(float), inSynInterOutput));
    }
}

void pushInterOutputStateToDevice(bool uninitialisedOnly) {
    pushinSynInterOutputToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullInputSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntInput, CL_TRUE, 0, 7 * sizeof(unsigned int), glbSpkCntInput));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkInput, CL_TRUE, 0, 3500 * sizeof(unsigned int), glbSpkInput));
}

void pullInputCurrentSpikesFromDevice() {
}

void pullVInputFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_VInput, CL_TRUE, 0, 500 * sizeof(scalar), VInput));
}

void pullCurrentVInputFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_VInput, CL_TRUE, 0, 500 * sizeof(scalar), VInput));
}

void pullUInputFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_UInput, CL_TRUE, 0, 500 * sizeof(scalar), UInput));
}

void pullCurrentUInputFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_UInput, CL_TRUE, 0, 500 * sizeof(scalar), UInput));
}

void pullInputStateFromDevice() {
    pullVInputFromDevice();
    pullUInputFromDevice();
}

void pullInputCurrentSourceStateFromDevice() {
}

void pullInterSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntInter, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntInter));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkInter, CL_TRUE, 0, 500 * sizeof(unsigned int), glbSpkInter));
}

void pullInterCurrentSpikesFromDevice() {
}

void pullVInterFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_VInter, CL_TRUE, 0, 500 * sizeof(scalar), VInter));
}

void pullCurrentVInterFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_VInter, CL_TRUE, 0, 500 * sizeof(scalar), VInter));
}

void pullUInterFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_UInter, CL_TRUE, 0, 500 * sizeof(scalar), UInter));
}

void pullCurrentUInterFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_UInter, CL_TRUE, 0, 500 * sizeof(scalar), UInter));
}

void pullInterStateFromDevice() {
    pullVInterFromDevice();
    pullUInterFromDevice();
}

void pullOutputSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntOutput, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntOutput));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkOutput, CL_TRUE, 0, 500 * sizeof(unsigned int), glbSpkOutput));
}

void pullOutputCurrentSpikesFromDevice() {
}

void pullVOutputFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_VOutput, CL_TRUE, 0, 500 * sizeof(scalar), VOutput));
}

void pullCurrentVOutputFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_VOutput, CL_TRUE, 0, 500 * sizeof(scalar), VOutput));
}

void pullUOutputFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_UOutput, CL_TRUE, 0, 500 * sizeof(scalar), UOutput));
}

void pullCurrentUOutputFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_UOutput, CL_TRUE, 0, 500 * sizeof(scalar), UOutput));
}

void pullOutputStateFromDevice() {
    pullVOutputFromDevice();
    pullUOutputFromDevice();
}

void pullinSynInputInterFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynInputInter, CL_TRUE, 0, 500 * sizeof(float), inSynInputInter));
}

void pullInputInterStateFromDevice() {
    pullinSynInputInterFromDevice();
}

void pullinSynInputOutputFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynInputOutput, CL_TRUE, 0, 500 * sizeof(float), inSynInputOutput));
}

void pullInputOutputStateFromDevice() {
    pullinSynInputOutputFromDevice();
}

void pullinSynInterOutputFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynInterOutput, CL_TRUE, 0, 500 * sizeof(float), inSynInterOutput));
}

void pullInterOutputStateFromDevice() {
    pullinSynInterOutputFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getInputCurrentSpikes() {
    return  (glbSpkInput + (spkQuePtrInput * 500));
}

unsigned int& getInputCurrentSpikeCount() {
    return glbSpkCntInput[spkQuePtrInput];
}

scalar* getCurrentVInput() {
    return VInput;
}

scalar* getCurrentUInput() {
    return UInput;
}

unsigned int* getInterCurrentSpikes() {
    return  glbSpkInter;
}

unsigned int& getInterCurrentSpikeCount() {
    return glbSpkCntInter[0];
}

scalar* getCurrentVInter() {
    return VInter;
}

scalar* getCurrentUInter() {
    return UInter;
}

unsigned int* getOutputCurrentSpikes() {
    return  glbSpkOutput;
}

unsigned int& getOutputCurrentSpikeCount() {
    return glbSpkCntOutput[0];
}

scalar* getCurrentVOutput() {
    return VOutput;
}

scalar* getCurrentUOutput() {
    return UOutput;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushInputStateToDevice(uninitialisedOnly);
    pushInterStateToDevice(uninitialisedOnly);
    pushOutputStateToDevice(uninitialisedOnly);
    pushInputCurrentSourceStateToDevice(uninitialisedOnly);
    pushInputInterStateToDevice(uninitialisedOnly);
    pushInputOutputStateToDevice(uninitialisedOnly);
    pushInterOutputStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
}

void copyStateFromDevice() {
    pullInputStateFromDevice();
    pullInterStateFromDevice();
    pullOutputStateFromDevice();
    pullInputCurrentSourceStateFromDevice();
    pullInputInterStateFromDevice();
    pullInputOutputStateFromDevice();
    pullInterOutputStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullInputCurrentSpikesFromDevice();
    pullInterCurrentSpikesFromDevice();
    pullOutputCurrentSpikesFromDevice();
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
    glbSpkCntInput = (unsigned int*)malloc(7 * sizeof(unsigned int));
    d_glbSpkCntInput = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 7 * sizeof (unsigned int), glbSpkCntInput);
    glbSpkInput = (unsigned int*)malloc(3500 * sizeof(unsigned int));
    d_glbSpkInput = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3500 * sizeof (unsigned int), glbSpkInput);
    VInput = (scalar*)malloc(500 * sizeof(scalar));
    d_VInput = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 500 * sizeof (scalar), VInput);
    UInput = (scalar*)malloc(500 * sizeof(scalar));
    d_UInput = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 500 * sizeof (scalar), UInput);
    // current source variables
    glbSpkCntInter = (unsigned int*)malloc(1 * sizeof(unsigned int));
    d_glbSpkCntInter = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof (unsigned int), glbSpkCntInter);
    glbSpkInter = (unsigned int*)malloc(500 * sizeof(unsigned int));
    d_glbSpkInter = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 500 * sizeof (unsigned int), glbSpkInter);
    VInter = (scalar*)malloc(500 * sizeof(scalar));
    d_VInter = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 500 * sizeof (scalar), VInter);
    UInter = (scalar*)malloc(500 * sizeof(scalar));
    d_UInter = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 500 * sizeof (scalar), UInter);
    glbSpkCntOutput = (unsigned int*)malloc(1 * sizeof(unsigned int));
    d_glbSpkCntOutput = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof (unsigned int), glbSpkCntOutput);
    glbSpkOutput = (unsigned int*)malloc(500 * sizeof(unsigned int));
    d_glbSpkOutput = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 500 * sizeof (unsigned int), glbSpkOutput);
    VOutput = (scalar*)malloc(500 * sizeof(scalar));
    d_VOutput = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 500 * sizeof (scalar), VOutput);
    UOutput = (scalar*)malloc(500 * sizeof(scalar));
    d_UOutput = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 500 * sizeof (scalar), UOutput);
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    inSynInputInter = (float*)malloc(500 * sizeof(float));
    d_inSynInputInter = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 500 * sizeof (float), inSynInputInter);
    inSynInterOutput = (float*)malloc(500 * sizeof(float));
    d_inSynInterOutput = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 500 * sizeof (float), inSynInterOutput);
    inSynInputOutput = (float*)malloc(500 * sizeof(float));
    d_inSynInputOutput = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 500 * sizeof (float), inSynInputOutput);
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
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
    free(glbSpkCntInput);
    free(glbSpkInput);
    free(VInput);
    free(UInput);
    // current source variables
    free(glbSpkCntInter);
    free(glbSpkInter);
    free(VInter);
    free(UInter);
    free(glbSpkCntOutput);
    free(glbSpkOutput);
    free(VOutput);
    free(UOutput);
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    free(inSynInputInter);
    free(inSynInterOutput);
    free(inSynInputOutput);
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
}

void stepTime() {
    updateSynapses(t);
    spkQuePtrInput = (spkQuePtrInput + 1) % 7;
    updateNeurons(t);
    iT++;
    t = iT*DT;
}

