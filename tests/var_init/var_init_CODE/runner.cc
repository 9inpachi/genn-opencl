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
    program.build("-cl-std=CL1.2 -I clRNG/include");
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
clrngLfsr113Stream* rng;
cl::Buffer d_rng;

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
unsigned int* glbSpkCntPop;
cl::Buffer d_glbSpkCntPop;
unsigned int* glbSpkPop;
cl::Buffer d_glbSpkPop;
scalar* constantPop;
cl::Buffer d_constantPop;
scalar* uniformPop;
cl::Buffer d_uniformPop;
scalar* normalPop;
cl::Buffer d_normalPop;
scalar* exponentialPop;
cl::Buffer d_exponentialPop;
scalar* gammaPop;
cl::Buffer d_gammaPop;
// current source variables
scalar* constantCurrSource;
cl::Buffer d_constantCurrSource;
scalar* uniformCurrSource;
cl::Buffer d_uniformCurrSource;
scalar* normalCurrSource;
cl::Buffer d_normalCurrSource;
scalar* exponentialCurrSource;
cl::Buffer d_exponentialCurrSource;
scalar* gammaCurrSource;
cl::Buffer d_gammaCurrSource;
unsigned int* glbSpkCntSpikeSource;
cl::Buffer d_glbSpkCntSpikeSource;
unsigned int* glbSpkSpikeSource;
cl::Buffer d_glbSpkSpikeSource;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
float* inSynSparse;
cl::Buffer d_inSynSparse;
scalar* pconstantSparse;
cl::Buffer d_pconstantSparse;
scalar* puniformSparse;
cl::Buffer d_puniformSparse;
scalar* pnormalSparse;
cl::Buffer d_pnormalSparse;
scalar* pexponentialSparse;
cl::Buffer d_pexponentialSparse;
scalar* pgammaSparse;
cl::Buffer d_pgammaSparse;
float* inSynDense;
cl::Buffer d_inSynDense;
scalar* pconstantDense;
cl::Buffer d_pconstantDense;
scalar* puniformDense;
cl::Buffer d_puniformDense;
scalar* pnormalDense;
cl::Buffer d_pnormalDense;
scalar* pexponentialDense;
cl::Buffer d_pexponentialDense;
scalar* pgammaDense;
cl::Buffer d_pgammaDense;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthSparse = 10000;
unsigned int* rowLengthSparse;
cl::Buffer d_rowLengthSparse;
uint32_t* indSparse;
cl::Buffer d_indSparse;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
scalar* constantDense;
cl::Buffer d_constantDense;
scalar* uniformDense;
cl::Buffer d_uniformDense;
scalar* normalDense;
cl::Buffer d_normalDense;
scalar* exponentialDense;
cl::Buffer d_exponentialDense;
scalar* gammaDense;
cl::Buffer d_gammaDense;
scalar* constantSparse;
cl::Buffer d_constantSparse;
scalar* uniformSparse;
cl::Buffer d_uniformSparse;
scalar* normalSparse;
cl::Buffer d_normalSparse;
scalar* exponentialSparse;
cl::Buffer d_exponentialSparse;
scalar* gammaSparse;
cl::Buffer d_gammaSparse;

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushPopSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntPop, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntPop));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkPop, CL_TRUE, 0, 10000 * sizeof(unsigned int), glbSpkPop));
    }
}

void pushPopCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntPop, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntPop));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkPop, CL_TRUE, 0, glbSpkCntPop[0] * sizeof(unsigned int), glbSpkPop));
}

void pushconstantPopToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_constantPop, CL_TRUE, 0, 10000 * sizeof(scalar), constantPop));
    }
}

void pushCurrentconstantPopToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_constantPop, CL_TRUE, 0, 10000 * sizeof(scalar), constantPop));
}

void pushuniformPopToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_uniformPop, CL_TRUE, 0, 10000 * sizeof(scalar), uniformPop));
    }
}

void pushCurrentuniformPopToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_uniformPop, CL_TRUE, 0, 10000 * sizeof(scalar), uniformPop));
}

void pushnormalPopToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_normalPop, CL_TRUE, 0, 10000 * sizeof(scalar), normalPop));
    }
}

void pushCurrentnormalPopToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_normalPop, CL_TRUE, 0, 10000 * sizeof(scalar), normalPop));
}

void pushexponentialPopToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_exponentialPop, CL_TRUE, 0, 10000 * sizeof(scalar), exponentialPop));
    }
}

void pushCurrentexponentialPopToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_exponentialPop, CL_TRUE, 0, 10000 * sizeof(scalar), exponentialPop));
}

void pushgammaPopToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_gammaPop, CL_TRUE, 0, 10000 * sizeof(scalar), gammaPop));
    }
}

void pushCurrentgammaPopToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_gammaPop, CL_TRUE, 0, 10000 * sizeof(scalar), gammaPop));
}

void pushPopStateToDevice(bool uninitialisedOnly) {
    pushconstantPopToDevice(uninitialisedOnly);
    pushuniformPopToDevice(uninitialisedOnly);
    pushnormalPopToDevice(uninitialisedOnly);
    pushexponentialPopToDevice(uninitialisedOnly);
    pushgammaPopToDevice(uninitialisedOnly);
}

void pushconstantCurrSourceToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_constantCurrSource, CL_TRUE, 0, 10000 * sizeof(scalar), constantCurrSource));
    }
}

void pushuniformCurrSourceToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_uniformCurrSource, CL_TRUE, 0, 10000 * sizeof(scalar), uniformCurrSource));
    }
}

void pushnormalCurrSourceToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_normalCurrSource, CL_TRUE, 0, 10000 * sizeof(scalar), normalCurrSource));
    }
}

void pushexponentialCurrSourceToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_exponentialCurrSource, CL_TRUE, 0, 10000 * sizeof(scalar), exponentialCurrSource));
    }
}

void pushgammaCurrSourceToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_gammaCurrSource, CL_TRUE, 0, 10000 * sizeof(scalar), gammaCurrSource));
    }
}

void pushCurrSourceStateToDevice(bool uninitialisedOnly) {
    pushconstantCurrSourceToDevice(uninitialisedOnly);
    pushuniformCurrSourceToDevice(uninitialisedOnly);
    pushnormalCurrSourceToDevice(uninitialisedOnly);
    pushexponentialCurrSourceToDevice(uninitialisedOnly);
    pushgammaCurrSourceToDevice(uninitialisedOnly);
}

void pushSpikeSourceSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntSpikeSource, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntSpikeSource));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkSpikeSource, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkSpikeSource));
    }
}

void pushSpikeSourceCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntSpikeSource, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntSpikeSource));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkSpikeSource, CL_TRUE, 0, glbSpkCntSpikeSource[0] * sizeof(unsigned int), glbSpkSpikeSource));
}

void pushSpikeSourceStateToDevice(bool uninitialisedOnly) {
}

void pushSparseConnectivityToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_rowLengthSparse, CL_TRUE, 0, 1 * sizeof(unsigned int), rowLengthSparse));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_indSparse, CL_TRUE, 0, 10000 * sizeof(unsigned int), indSparse));
}

void pushconstantDenseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_constantDense, CL_TRUE, 0, 10000 * sizeof(scalar), constantDense));
    }
}

void pushuniformDenseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_uniformDense, CL_TRUE, 0, 10000 * sizeof(scalar), uniformDense));
    }
}

void pushnormalDenseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_normalDense, CL_TRUE, 0, 10000 * sizeof(scalar), normalDense));
    }
}

void pushexponentialDenseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_exponentialDense, CL_TRUE, 0, 10000 * sizeof(scalar), exponentialDense));
    }
}

void pushgammaDenseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_gammaDense, CL_TRUE, 0, 10000 * sizeof(scalar), gammaDense));
    }
}

void pushinSynDenseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynDense, CL_TRUE, 0, 10000 * sizeof(float), inSynDense));
    }
}

void pushpconstantDenseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_pconstantDense, CL_TRUE, 0, 10000 * sizeof(scalar), pconstantDense));
    }
}

void pushpuniformDenseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_puniformDense, CL_TRUE, 0, 10000 * sizeof(scalar), puniformDense));
    }
}

void pushpnormalDenseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_pnormalDense, CL_TRUE, 0, 10000 * sizeof(scalar), pnormalDense));
    }
}

void pushpexponentialDenseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_pexponentialDense, CL_TRUE, 0, 10000 * sizeof(scalar), pexponentialDense));
    }
}

void pushpgammaDenseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_pgammaDense, CL_TRUE, 0, 10000 * sizeof(scalar), pgammaDense));
    }
}

void pushDenseStateToDevice(bool uninitialisedOnly) {
    pushconstantDenseToDevice(uninitialisedOnly);
    pushuniformDenseToDevice(uninitialisedOnly);
    pushnormalDenseToDevice(uninitialisedOnly);
    pushexponentialDenseToDevice(uninitialisedOnly);
    pushgammaDenseToDevice(uninitialisedOnly);
    pushinSynDenseToDevice(uninitialisedOnly);
    pushpconstantDenseToDevice(uninitialisedOnly);
    pushpuniformDenseToDevice(uninitialisedOnly);
    pushpnormalDenseToDevice(uninitialisedOnly);
    pushpexponentialDenseToDevice(uninitialisedOnly);
    pushpgammaDenseToDevice(uninitialisedOnly);
}

void pushconstantSparseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_constantSparse, CL_TRUE, 0, 10000 * sizeof(scalar), constantSparse));
    }
}

void pushuniformSparseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_uniformSparse, CL_TRUE, 0, 10000 * sizeof(scalar), uniformSparse));
    }
}

void pushnormalSparseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_normalSparse, CL_TRUE, 0, 10000 * sizeof(scalar), normalSparse));
    }
}

void pushexponentialSparseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_exponentialSparse, CL_TRUE, 0, 10000 * sizeof(scalar), exponentialSparse));
    }
}

void pushgammaSparseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_gammaSparse, CL_TRUE, 0, 10000 * sizeof(scalar), gammaSparse));
    }
}

void pushinSynSparseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynSparse, CL_TRUE, 0, 10000 * sizeof(float), inSynSparse));
    }
}

void pushpconstantSparseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_pconstantSparse, CL_TRUE, 0, 10000 * sizeof(scalar), pconstantSparse));
    }
}

void pushpuniformSparseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_puniformSparse, CL_TRUE, 0, 10000 * sizeof(scalar), puniformSparse));
    }
}

void pushpnormalSparseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_pnormalSparse, CL_TRUE, 0, 10000 * sizeof(scalar), pnormalSparse));
    }
}

void pushpexponentialSparseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_pexponentialSparse, CL_TRUE, 0, 10000 * sizeof(scalar), pexponentialSparse));
    }
}

void pushpgammaSparseToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_pgammaSparse, CL_TRUE, 0, 10000 * sizeof(scalar), pgammaSparse));
    }
}

void pushSparseStateToDevice(bool uninitialisedOnly) {
    pushconstantSparseToDevice(uninitialisedOnly);
    pushuniformSparseToDevice(uninitialisedOnly);
    pushnormalSparseToDevice(uninitialisedOnly);
    pushexponentialSparseToDevice(uninitialisedOnly);
    pushgammaSparseToDevice(uninitialisedOnly);
    pushinSynSparseToDevice(uninitialisedOnly);
    pushpconstantSparseToDevice(uninitialisedOnly);
    pushpuniformSparseToDevice(uninitialisedOnly);
    pushpnormalSparseToDevice(uninitialisedOnly);
    pushpexponentialSparseToDevice(uninitialisedOnly);
    pushpgammaSparseToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullPopSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntPop, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntPop));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkPop, CL_TRUE, 0, 10000 * sizeof(unsigned int), glbSpkPop));
}

void pullPopCurrentSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntPop, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntPop));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkPop, CL_TRUE, 0, glbSpkCntPop[0] * sizeof(unsigned int), glbSpkPop));
}

void pullconstantPopFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_constantPop, CL_TRUE, 0, 10000 * sizeof(scalar), constantPop));
}

void pullCurrentconstantPopFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_constantPop, CL_TRUE, 0, 10000 * sizeof(scalar), constantPop));
}

void pulluniformPopFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_uniformPop, CL_TRUE, 0, 10000 * sizeof(scalar), uniformPop));
}

void pullCurrentuniformPopFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_uniformPop, CL_TRUE, 0, 10000 * sizeof(scalar), uniformPop));
}

void pullnormalPopFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_normalPop, CL_TRUE, 0, 10000 * sizeof(scalar), normalPop));
}

void pullCurrentnormalPopFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_normalPop, CL_TRUE, 0, 10000 * sizeof(scalar), normalPop));
}

void pullexponentialPopFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_exponentialPop, CL_TRUE, 0, 10000 * sizeof(scalar), exponentialPop));
}

void pullCurrentexponentialPopFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_exponentialPop, CL_TRUE, 0, 10000 * sizeof(scalar), exponentialPop));
}

void pullgammaPopFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_gammaPop, CL_TRUE, 0, 10000 * sizeof(scalar), gammaPop));
}

void pullCurrentgammaPopFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_gammaPop, CL_TRUE, 0, 10000 * sizeof(scalar), gammaPop));
}

void pullPopStateFromDevice() {
    pullconstantPopFromDevice();
    pulluniformPopFromDevice();
    pullnormalPopFromDevice();
    pullexponentialPopFromDevice();
    pullgammaPopFromDevice();
}

void pullconstantCurrSourceFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_constantCurrSource, CL_TRUE, 0, 10000 * sizeof(scalar), constantCurrSource));
}

void pulluniformCurrSourceFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_uniformCurrSource, CL_TRUE, 0, 10000 * sizeof(scalar), uniformCurrSource));
}

void pullnormalCurrSourceFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_normalCurrSource, CL_TRUE, 0, 10000 * sizeof(scalar), normalCurrSource));
}

void pullexponentialCurrSourceFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_exponentialCurrSource, CL_TRUE, 0, 10000 * sizeof(scalar), exponentialCurrSource));
}

void pullgammaCurrSourceFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_gammaCurrSource, CL_TRUE, 0, 10000 * sizeof(scalar), gammaCurrSource));
}

void pullCurrSourceStateFromDevice() {
    pullconstantCurrSourceFromDevice();
    pulluniformCurrSourceFromDevice();
    pullnormalCurrSourceFromDevice();
    pullexponentialCurrSourceFromDevice();
    pullgammaCurrSourceFromDevice();
}

void pullSpikeSourceSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntSpikeSource, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntSpikeSource));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkSpikeSource, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkSpikeSource));
}

void pullSpikeSourceCurrentSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntSpikeSource, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntSpikeSource));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkSpikeSource, CL_TRUE, 0, glbSpkCntSpikeSource[0] * sizeof(unsigned int), glbSpkSpikeSource));
}

void pullSpikeSourceStateFromDevice() {
}

void pullSparseConnectivityFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_rowLengthSparse, CL_TRUE, 0, 1 * sizeof(unsigned int), rowLengthSparse));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_indSparse, CL_TRUE, 0, 10000 * sizeof(unsigned int), indSparse));
}

void pullconstantDenseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_constantDense, CL_TRUE, 0, 10000 * sizeof(scalar), constantDense));
}

void pulluniformDenseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_uniformDense, CL_TRUE, 0, 10000 * sizeof(scalar), uniformDense));
}

void pullnormalDenseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_normalDense, CL_TRUE, 0, 10000 * sizeof(scalar), normalDense));
}

void pullexponentialDenseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_exponentialDense, CL_TRUE, 0, 10000 * sizeof(scalar), exponentialDense));
}

void pullgammaDenseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_gammaDense, CL_TRUE, 0, 10000 * sizeof(scalar), gammaDense));
}

void pullinSynDenseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynDense, CL_TRUE, 0, 10000 * sizeof(float), inSynDense));
}

void pullpconstantDenseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_pconstantDense, CL_TRUE, 0, 10000 * sizeof(scalar), pconstantDense));
}

void pullpuniformDenseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_puniformDense, CL_TRUE, 0, 10000 * sizeof(scalar), puniformDense));
}

void pullpnormalDenseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_pnormalDense, CL_TRUE, 0, 10000 * sizeof(scalar), pnormalDense));
}

void pullpexponentialDenseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_pexponentialDense, CL_TRUE, 0, 10000 * sizeof(scalar), pexponentialDense));
}

void pullpgammaDenseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_pgammaDense, CL_TRUE, 0, 10000 * sizeof(scalar), pgammaDense));
}

void pullDenseStateFromDevice() {
    pullconstantDenseFromDevice();
    pulluniformDenseFromDevice();
    pullnormalDenseFromDevice();
    pullexponentialDenseFromDevice();
    pullgammaDenseFromDevice();
    pullinSynDenseFromDevice();
    pullpconstantDenseFromDevice();
    pullpuniformDenseFromDevice();
    pullpnormalDenseFromDevice();
    pullpexponentialDenseFromDevice();
    pullpgammaDenseFromDevice();
}

void pullconstantSparseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_constantSparse, CL_TRUE, 0, 10000 * sizeof(scalar), constantSparse));
}

void pulluniformSparseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_uniformSparse, CL_TRUE, 0, 10000 * sizeof(scalar), uniformSparse));
}

void pullnormalSparseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_normalSparse, CL_TRUE, 0, 10000 * sizeof(scalar), normalSparse));
}

void pullexponentialSparseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_exponentialSparse, CL_TRUE, 0, 10000 * sizeof(scalar), exponentialSparse));
}

void pullgammaSparseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_gammaSparse, CL_TRUE, 0, 10000 * sizeof(scalar), gammaSparse));
}

void pullinSynSparseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynSparse, CL_TRUE, 0, 10000 * sizeof(float), inSynSparse));
}

void pullpconstantSparseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_pconstantSparse, CL_TRUE, 0, 10000 * sizeof(scalar), pconstantSparse));
}

void pullpuniformSparseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_puniformSparse, CL_TRUE, 0, 10000 * sizeof(scalar), puniformSparse));
}

void pullpnormalSparseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_pnormalSparse, CL_TRUE, 0, 10000 * sizeof(scalar), pnormalSparse));
}

void pullpexponentialSparseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_pexponentialSparse, CL_TRUE, 0, 10000 * sizeof(scalar), pexponentialSparse));
}

void pullpgammaSparseFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_pgammaSparse, CL_TRUE, 0, 10000 * sizeof(scalar), pgammaSparse));
}

void pullSparseStateFromDevice() {
    pullconstantSparseFromDevice();
    pulluniformSparseFromDevice();
    pullnormalSparseFromDevice();
    pullexponentialSparseFromDevice();
    pullgammaSparseFromDevice();
    pullinSynSparseFromDevice();
    pullpconstantSparseFromDevice();
    pullpuniformSparseFromDevice();
    pullpnormalSparseFromDevice();
    pullpexponentialSparseFromDevice();
    pullpgammaSparseFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getPopCurrentSpikes() {
    return  glbSpkPop;
}

unsigned int& getPopCurrentSpikeCount() {
    return glbSpkCntPop[0];
}

scalar* getCurrentconstantPop() {
    return constantPop;
}

scalar* getCurrentuniformPop() {
    return uniformPop;
}

scalar* getCurrentnormalPop() {
    return normalPop;
}

scalar* getCurrentexponentialPop() {
    return exponentialPop;
}

scalar* getCurrentgammaPop() {
    return gammaPop;
}

unsigned int* getSpikeSourceCurrentSpikes() {
    return  glbSpkSpikeSource;
}

unsigned int& getSpikeSourceCurrentSpikeCount() {
    return glbSpkCntSpikeSource[0];
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushPopStateToDevice(uninitialisedOnly);
    pushSpikeSourceStateToDevice(uninitialisedOnly);
    pushCurrSourceStateToDevice(uninitialisedOnly);
    pushDenseStateToDevice(uninitialisedOnly);
    pushSparseStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushSparseConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pullPopStateFromDevice();
    pullSpikeSourceStateFromDevice();
    pullCurrSourceStateFromDevice();
    pullDenseStateFromDevice();
    pullSparseStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullPopCurrentSpikesFromDevice();
    pullSpikeSourceCurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
}

void allocateMem() {
    initPrograms();
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    size_t rngCount = 1;
    rng = clrngLfsr113CreateStreams(NULL, 1, &rngCount, NULL);
    d_rng = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, rngCount, rng);
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // remote neuron groups
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    glbSpkCntPop = (unsigned int*)calloc(1, sizeof(unsigned int));
    d_glbSpkCntPop = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof(unsigned int), glbSpkCntPop);
    glbSpkPop = (unsigned int*)calloc(10000, sizeof(unsigned int));
    d_glbSpkPop = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(unsigned int), glbSpkPop);
    constantPop = (scalar*)calloc(10000, sizeof(scalar));
    d_constantPop = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), constantPop);
    uniformPop = (scalar*)calloc(10000, sizeof(scalar));
    d_uniformPop = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), uniformPop);
    normalPop = (scalar*)calloc(10000, sizeof(scalar));
    d_normalPop = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), normalPop);
    exponentialPop = (scalar*)calloc(10000, sizeof(scalar));
    d_exponentialPop = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), exponentialPop);
    gammaPop = (scalar*)calloc(10000, sizeof(scalar));
    d_gammaPop = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), gammaPop);
    // current source variables
    constantCurrSource = (scalar*)calloc(10000, sizeof(scalar));
    d_constantCurrSource = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), constantCurrSource);
    uniformCurrSource = (scalar*)calloc(10000, sizeof(scalar));
    d_uniformCurrSource = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), uniformCurrSource);
    normalCurrSource = (scalar*)calloc(10000, sizeof(scalar));
    d_normalCurrSource = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), normalCurrSource);
    exponentialCurrSource = (scalar*)calloc(10000, sizeof(scalar));
    d_exponentialCurrSource = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), exponentialCurrSource);
    gammaCurrSource = (scalar*)calloc(10000, sizeof(scalar));
    d_gammaCurrSource = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), gammaCurrSource);
    glbSpkCntSpikeSource = (unsigned int*)calloc(1, sizeof(unsigned int));
    d_glbSpkCntSpikeSource = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof(unsigned int), glbSpkCntSpikeSource);
    glbSpkSpikeSource = (unsigned int*)calloc(1, sizeof(unsigned int));
    d_glbSpkSpikeSource = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof(unsigned int), glbSpkSpikeSource);
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    inSynSparse = (float*)calloc(10000, sizeof(float));
    d_inSynSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(float), inSynSparse);
    pconstantSparse = (scalar*)calloc(10000, sizeof(scalar));
    d_pconstantSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), pconstantSparse);
    puniformSparse = (scalar*)calloc(10000, sizeof(scalar));
    d_puniformSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), puniformSparse);
    pnormalSparse = (scalar*)calloc(10000, sizeof(scalar));
    d_pnormalSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), pnormalSparse);
    pexponentialSparse = (scalar*)calloc(10000, sizeof(scalar));
    d_pexponentialSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), pexponentialSparse);
    pgammaSparse = (scalar*)calloc(10000, sizeof(scalar));
    d_pgammaSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), pgammaSparse);
    inSynDense = (float*)calloc(10000, sizeof(float));
    d_inSynDense = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(float), inSynDense);
    pconstantDense = (scalar*)calloc(10000, sizeof(scalar));
    d_pconstantDense = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), pconstantDense);
    puniformDense = (scalar*)calloc(10000, sizeof(scalar));
    d_puniformDense = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), puniformDense);
    pnormalDense = (scalar*)calloc(10000, sizeof(scalar));
    d_pnormalDense = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), pnormalDense);
    pexponentialDense = (scalar*)calloc(10000, sizeof(scalar));
    d_pexponentialDense = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), pexponentialDense);
    pgammaDense = (scalar*)calloc(10000, sizeof(scalar));
    d_pgammaDense = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), pgammaDense);
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    rowLengthSparse = (unsigned int*)calloc(1, sizeof(unsigned int));
    d_rowLengthSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof(unsigned int), rowLengthSparse);
    indSparse = (uint32_t*)calloc(10000, sizeof(uint32_t));
    d_indSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(uint32_t), indSparse);
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    constantDense = (scalar*)calloc(10000, sizeof(scalar));
    d_constantDense = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), constantDense);
    uniformDense = (scalar*)calloc(10000, sizeof(scalar));
    d_uniformDense = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), uniformDense);
    normalDense = (scalar*)calloc(10000, sizeof(scalar));
    d_normalDense = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), normalDense);
    exponentialDense = (scalar*)calloc(10000, sizeof(scalar));
    d_exponentialDense = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), exponentialDense);
    gammaDense = (scalar*)calloc(10000, sizeof(scalar));
    d_gammaDense = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), gammaDense);
    constantSparse = (scalar*)calloc(10000, sizeof(scalar));
    d_constantSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), constantSparse);
    uniformSparse = (scalar*)calloc(10000, sizeof(scalar));
    d_uniformSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), uniformSparse);
    normalSparse = (scalar*)calloc(10000, sizeof(scalar));
    d_normalSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), normalSparse);
    exponentialSparse = (scalar*)calloc(10000, sizeof(scalar));
    d_exponentialSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), exponentialSparse);
    gammaSparse = (scalar*)calloc(10000, sizeof(scalar));
    d_gammaSparse = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10000 * sizeof(scalar), gammaSparse);
    
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
    free(glbSpkCntPop);
    free(glbSpkPop);
    free(constantPop);
    free(uniformPop);
    free(normalPop);
    free(exponentialPop);
    free(gammaPop);
    // current source variables
    free(constantCurrSource);
    free(uniformCurrSource);
    free(normalCurrSource);
    free(exponentialCurrSource);
    free(gammaCurrSource);
    free(glbSpkCntSpikeSource);
    free(glbSpkSpikeSource);
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    free(inSynSparse);
    free(pconstantSparse);
    free(puniformSparse);
    free(pnormalSparse);
    free(pexponentialSparse);
    free(pgammaSparse);
    free(inSynDense);
    free(pconstantDense);
    free(puniformDense);
    free(pnormalDense);
    free(pexponentialDense);
    free(pgammaDense);
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    free(rowLengthSparse);
    free(indSparse);
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    free(constantDense);
    free(uniformDense);
    free(normalDense);
    free(exponentialDense);
    free(gammaDense);
    free(constantSparse);
    free(uniformSparse);
    free(normalSparse);
    free(exponentialSparse);
    free(gammaSparse);
    
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t);
    iT++;
    t = iT*DT;
}

