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
scalar* xpost;
cl::Buffer d_xpost;
scalar* shiftpost;
cl::Buffer d_shiftpost;
unsigned int* glbSpkCntpre;
cl::Buffer d_glbSpkCntpre;
unsigned int* glbSpkpre;
cl::Buffer d_glbSpkpre;
unsigned int spkQuePtrpre;
cl::Buffer d_spkQuePtrpre;
scalar* xpre;
cl::Buffer d_xpre;
scalar* shiftpre;
cl::Buffer d_shiftpre;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
float* inSynsyn9;
cl::Buffer d_inSynsyn9;
float* inSynsyn8;
cl::Buffer d_inSynsyn8;
float* inSynsyn7;
cl::Buffer d_inSynsyn7;
float* inSynsyn6;
cl::Buffer d_inSynsyn6;
float* inSynsyn5;
cl::Buffer d_inSynsyn5;
float* inSynsyn4;
cl::Buffer d_inSynsyn4;
float* inSynsyn3;
cl::Buffer d_inSynsyn3;
float* inSynsyn2;
cl::Buffer d_inSynsyn2;
float* inSynsyn1;
cl::Buffer d_inSynsyn1;
float* inSynsyn0;
cl::Buffer d_inSynsyn0;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
scalar* wsyn0;
cl::Buffer d_wsyn0;
scalar* wsyn1;
cl::Buffer d_wsyn1;
scalar* wsyn2;
cl::Buffer d_wsyn2;
scalar* wsyn3;
cl::Buffer d_wsyn3;
scalar* wsyn4;
cl::Buffer d_wsyn4;
scalar* wsyn5;
cl::Buffer d_wsyn5;
scalar* wsyn6;
cl::Buffer d_wsyn6;
scalar* wsyn7;
cl::Buffer d_wsyn7;
scalar* wsyn8;
cl::Buffer d_wsyn8;
scalar* wsyn9;
cl::Buffer d_wsyn9;

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushpostSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntpost, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntpost));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkpost, CL_TRUE, 0, 10 * sizeof(unsigned int), glbSpkpost));
    }
}

void pushpostCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntpost, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntpost));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkpost, CL_TRUE, 0, glbSpkCntpost[0] * sizeof(unsigned int), glbSpkpost));
}

void pushxpostToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_xpost, CL_TRUE, 0, 10 * sizeof(scalar), xpost));
    }
}

void pushCurrentxpostToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_xpost, CL_TRUE, 0, 10 * sizeof(scalar), xpost));
}

void pushshiftpostToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_shiftpost, CL_TRUE, 0, 10 * sizeof(scalar), shiftpost));
}

void pushCurrentshiftpostToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_shiftpost, CL_TRUE, 0, 10 * sizeof(scalar), shiftpost));
}

void pushpostStateToDevice(bool uninitialisedOnly) {
    pushxpostToDevice(uninitialisedOnly);
    pushshiftpostToDevice(uninitialisedOnly);
}

void pushpreSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntpre, CL_TRUE, 0, 10 * sizeof(unsigned int), glbSpkCntpre));
    }
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkpre, CL_TRUE, 0, 100 * sizeof(unsigned int), glbSpkpre));
    }
}

void pushpreCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkCntpre, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntpre));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_glbSpkpre, CL_TRUE, 0, 10 * sizeof(unsigned int), glbSpkpre));
}

void pushxpreToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_xpre, CL_TRUE, 0, 100 * sizeof(scalar), xpre));
    }
}

void pushCurrentxpreToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_xpre, CL_TRUE, 0, 10 * sizeof(scalar), &xpre[spkQuePtrpre * 10]));
}

void pushshiftpreToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_shiftpre, CL_TRUE, 0, 10 * sizeof(scalar), shiftpre));
}

void pushCurrentshiftpreToDevice(bool uninitialisedOnly) {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_shiftpre, CL_TRUE, 0, 10 * sizeof(scalar), shiftpre));
}

void pushpreStateToDevice(bool uninitialisedOnly) {
    pushxpreToDevice(uninitialisedOnly);
    pushshiftpreToDevice(uninitialisedOnly);
}

void pushwsyn0ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_wsyn0, CL_TRUE, 0, 100 * sizeof(scalar), wsyn0));
    }
}

void pushinSynsyn0ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynsyn0, CL_TRUE, 0, 10 * sizeof(float), inSynsyn0));
    }
}

void pushsyn0StateToDevice(bool uninitialisedOnly) {
    pushwsyn0ToDevice(uninitialisedOnly);
    pushinSynsyn0ToDevice(uninitialisedOnly);
}

void pushwsyn1ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_wsyn1, CL_TRUE, 0, 100 * sizeof(scalar), wsyn1));
    }
}

void pushinSynsyn1ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynsyn1, CL_TRUE, 0, 10 * sizeof(float), inSynsyn1));
    }
}

void pushsyn1StateToDevice(bool uninitialisedOnly) {
    pushwsyn1ToDevice(uninitialisedOnly);
    pushinSynsyn1ToDevice(uninitialisedOnly);
}

void pushwsyn2ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_wsyn2, CL_TRUE, 0, 100 * sizeof(scalar), wsyn2));
    }
}

void pushinSynsyn2ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynsyn2, CL_TRUE, 0, 10 * sizeof(float), inSynsyn2));
    }
}

void pushsyn2StateToDevice(bool uninitialisedOnly) {
    pushwsyn2ToDevice(uninitialisedOnly);
    pushinSynsyn2ToDevice(uninitialisedOnly);
}

void pushwsyn3ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_wsyn3, CL_TRUE, 0, 100 * sizeof(scalar), wsyn3));
    }
}

void pushinSynsyn3ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynsyn3, CL_TRUE, 0, 10 * sizeof(float), inSynsyn3));
    }
}

void pushsyn3StateToDevice(bool uninitialisedOnly) {
    pushwsyn3ToDevice(uninitialisedOnly);
    pushinSynsyn3ToDevice(uninitialisedOnly);
}

void pushwsyn4ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_wsyn4, CL_TRUE, 0, 100 * sizeof(scalar), wsyn4));
    }
}

void pushinSynsyn4ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynsyn4, CL_TRUE, 0, 10 * sizeof(float), inSynsyn4));
    }
}

void pushsyn4StateToDevice(bool uninitialisedOnly) {
    pushwsyn4ToDevice(uninitialisedOnly);
    pushinSynsyn4ToDevice(uninitialisedOnly);
}

void pushwsyn5ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_wsyn5, CL_TRUE, 0, 100 * sizeof(scalar), wsyn5));
    }
}

void pushinSynsyn5ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynsyn5, CL_TRUE, 0, 10 * sizeof(float), inSynsyn5));
    }
}

void pushsyn5StateToDevice(bool uninitialisedOnly) {
    pushwsyn5ToDevice(uninitialisedOnly);
    pushinSynsyn5ToDevice(uninitialisedOnly);
}

void pushwsyn6ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_wsyn6, CL_TRUE, 0, 100 * sizeof(scalar), wsyn6));
    }
}

void pushinSynsyn6ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynsyn6, CL_TRUE, 0, 10 * sizeof(float), inSynsyn6));
    }
}

void pushsyn6StateToDevice(bool uninitialisedOnly) {
    pushwsyn6ToDevice(uninitialisedOnly);
    pushinSynsyn6ToDevice(uninitialisedOnly);
}

void pushwsyn7ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_wsyn7, CL_TRUE, 0, 100 * sizeof(scalar), wsyn7));
    }
}

void pushinSynsyn7ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynsyn7, CL_TRUE, 0, 10 * sizeof(float), inSynsyn7));
    }
}

void pushsyn7StateToDevice(bool uninitialisedOnly) {
    pushwsyn7ToDevice(uninitialisedOnly);
    pushinSynsyn7ToDevice(uninitialisedOnly);
}

void pushwsyn8ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_wsyn8, CL_TRUE, 0, 100 * sizeof(scalar), wsyn8));
    }
}

void pushinSynsyn8ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynsyn8, CL_TRUE, 0, 10 * sizeof(float), inSynsyn8));
    }
}

void pushsyn8StateToDevice(bool uninitialisedOnly) {
    pushwsyn8ToDevice(uninitialisedOnly);
    pushinSynsyn8ToDevice(uninitialisedOnly);
}

void pushwsyn9ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_wsyn9, CL_TRUE, 0, 100 * sizeof(scalar), wsyn9));
    }
}

void pushinSynsyn9ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_inSynsyn9, CL_TRUE, 0, 10 * sizeof(float), inSynsyn9));
    }
}

void pushsyn9StateToDevice(bool uninitialisedOnly) {
    pushwsyn9ToDevice(uninitialisedOnly);
    pushinSynsyn9ToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullpostSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntpost, CL_TRUE, 0, 1 * sizeof(unsigned int), glbSpkCntpost));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkpost, CL_TRUE, 0, 10 * sizeof(unsigned int), glbSpkpost));
}

void pullpostCurrentSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntpost, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntpost));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkpost, CL_TRUE, 0, glbSpkCntpost[0] * sizeof(unsigned int), glbSpkpost));
}

void pullxpostFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_xpost, CL_TRUE, 0, 10 * sizeof(scalar), xpost));
}

void pullCurrentxpostFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_xpost, CL_TRUE, 0, 10 * sizeof(scalar), xpost));
}

void pullshiftpostFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_shiftpost, CL_TRUE, 0, 10 * sizeof(scalar), shiftpost));
}

void pullCurrentshiftpostFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_shiftpost, CL_TRUE, 0, 10 * sizeof(scalar), shiftpost));
}

void pullpostStateFromDevice() {
    pullxpostFromDevice();
    pullshiftpostFromDevice();
}

void pullpreSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntpre, CL_TRUE, 0, 10 * sizeof(unsigned int), glbSpkCntpre));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkpre, CL_TRUE, 0, 100 * sizeof(unsigned int), glbSpkpre));
}

void pullpreCurrentSpikesFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkCntpre, CL_TRUE, 0, sizeof(unsigned int), glbSpkCntpre));
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_glbSpkpre, CL_TRUE, 0, 10 * sizeof(unsigned int), glbSpkpre));
}

void pullxpreFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_xpre, CL_TRUE, 0, 100 * sizeof(scalar), xpre));
}

void pullCurrentxpreFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_xpre, CL_TRUE, 0, 10 * sizeof(scalar), &xpre[spkQuePtrpre * 10]));
}

void pullshiftpreFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_shiftpre, CL_TRUE, 0, 10 * sizeof(scalar), shiftpre));
}

void pullCurrentshiftpreFromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_shiftpre, CL_TRUE, 0, 10 * sizeof(scalar), shiftpre));
}

void pullpreStateFromDevice() {
    pullxpreFromDevice();
    pullshiftpreFromDevice();
}

void pullwsyn0FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_wsyn0, CL_TRUE, 0, 100 * sizeof(scalar), wsyn0));
}

void pullinSynsyn0FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynsyn0, CL_TRUE, 0, 10 * sizeof(float), inSynsyn0));
}

void pullsyn0StateFromDevice() {
    pullwsyn0FromDevice();
    pullinSynsyn0FromDevice();
}

void pullwsyn1FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_wsyn1, CL_TRUE, 0, 100 * sizeof(scalar), wsyn1));
}

void pullinSynsyn1FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynsyn1, CL_TRUE, 0, 10 * sizeof(float), inSynsyn1));
}

void pullsyn1StateFromDevice() {
    pullwsyn1FromDevice();
    pullinSynsyn1FromDevice();
}

void pullwsyn2FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_wsyn2, CL_TRUE, 0, 100 * sizeof(scalar), wsyn2));
}

void pullinSynsyn2FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynsyn2, CL_TRUE, 0, 10 * sizeof(float), inSynsyn2));
}

void pullsyn2StateFromDevice() {
    pullwsyn2FromDevice();
    pullinSynsyn2FromDevice();
}

void pullwsyn3FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_wsyn3, CL_TRUE, 0, 100 * sizeof(scalar), wsyn3));
}

void pullinSynsyn3FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynsyn3, CL_TRUE, 0, 10 * sizeof(float), inSynsyn3));
}

void pullsyn3StateFromDevice() {
    pullwsyn3FromDevice();
    pullinSynsyn3FromDevice();
}

void pullwsyn4FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_wsyn4, CL_TRUE, 0, 100 * sizeof(scalar), wsyn4));
}

void pullinSynsyn4FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynsyn4, CL_TRUE, 0, 10 * sizeof(float), inSynsyn4));
}

void pullsyn4StateFromDevice() {
    pullwsyn4FromDevice();
    pullinSynsyn4FromDevice();
}

void pullwsyn5FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_wsyn5, CL_TRUE, 0, 100 * sizeof(scalar), wsyn5));
}

void pullinSynsyn5FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynsyn5, CL_TRUE, 0, 10 * sizeof(float), inSynsyn5));
}

void pullsyn5StateFromDevice() {
    pullwsyn5FromDevice();
    pullinSynsyn5FromDevice();
}

void pullwsyn6FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_wsyn6, CL_TRUE, 0, 100 * sizeof(scalar), wsyn6));
}

void pullinSynsyn6FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynsyn6, CL_TRUE, 0, 10 * sizeof(float), inSynsyn6));
}

void pullsyn6StateFromDevice() {
    pullwsyn6FromDevice();
    pullinSynsyn6FromDevice();
}

void pullwsyn7FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_wsyn7, CL_TRUE, 0, 100 * sizeof(scalar), wsyn7));
}

void pullinSynsyn7FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynsyn7, CL_TRUE, 0, 10 * sizeof(float), inSynsyn7));
}

void pullsyn7StateFromDevice() {
    pullwsyn7FromDevice();
    pullinSynsyn7FromDevice();
}

void pullwsyn8FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_wsyn8, CL_TRUE, 0, 100 * sizeof(scalar), wsyn8));
}

void pullinSynsyn8FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynsyn8, CL_TRUE, 0, 10 * sizeof(float), inSynsyn8));
}

void pullsyn8StateFromDevice() {
    pullwsyn8FromDevice();
    pullinSynsyn8FromDevice();
}

void pullwsyn9FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_wsyn9, CL_TRUE, 0, 100 * sizeof(scalar), wsyn9));
}

void pullinSynsyn9FromDevice() {
    CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_inSynsyn9, CL_TRUE, 0, 10 * sizeof(float), inSynsyn9));
}

void pullsyn9StateFromDevice() {
    pullwsyn9FromDevice();
    pullinSynsyn9FromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getpostCurrentSpikes() {
    return  glbSpkpost;
}

unsigned int& getpostCurrentSpikeCount() {
    return glbSpkCntpost[0];
}

scalar* getCurrentxpost() {
    return xpost;
}

scalar* getCurrentshiftpost() {
    return shiftpost;
}

unsigned int* getpreCurrentSpikes() {
    return  (glbSpkpre + (spkQuePtrpre * 10));
}

unsigned int& getpreCurrentSpikeCount() {
    return glbSpkCntpre[spkQuePtrpre];
}

scalar* getCurrentxpre() {
    return xpre + (spkQuePtrpre * 10);
}

scalar* getCurrentshiftpre() {
    return shiftpre;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushpostStateToDevice(uninitialisedOnly);
    pushpreStateToDevice(uninitialisedOnly);
    pushsyn0StateToDevice(uninitialisedOnly);
    pushsyn1StateToDevice(uninitialisedOnly);
    pushsyn2StateToDevice(uninitialisedOnly);
    pushsyn3StateToDevice(uninitialisedOnly);
    pushsyn4StateToDevice(uninitialisedOnly);
    pushsyn5StateToDevice(uninitialisedOnly);
    pushsyn6StateToDevice(uninitialisedOnly);
    pushsyn7StateToDevice(uninitialisedOnly);
    pushsyn8StateToDevice(uninitialisedOnly);
    pushsyn9StateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
}

void copyStateFromDevice() {
    pullpostStateFromDevice();
    pullpreStateFromDevice();
    pullsyn0StateFromDevice();
    pullsyn1StateFromDevice();
    pullsyn2StateFromDevice();
    pullsyn3StateFromDevice();
    pullsyn4StateFromDevice();
    pullsyn5StateFromDevice();
    pullsyn6StateFromDevice();
    pullsyn7StateFromDevice();
    pullsyn8StateFromDevice();
    pullsyn9StateFromDevice();
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
    glbSpkCntpost = (unsigned int*)calloc(1, sizeof(unsigned int));
    d_glbSpkCntpost = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof (unsigned int), glbSpkCntpost);
    glbSpkpost = (unsigned int*)calloc(10, sizeof(unsigned int));
    d_glbSpkpost = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (unsigned int), glbSpkpost);
    xpost = (scalar*)calloc(10, sizeof(scalar));
    d_xpost = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (scalar), xpost);
    shiftpost = (scalar*)calloc(10, sizeof(scalar));
    d_shiftpost = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (scalar), shiftpost);
    glbSpkCntpre = (unsigned int*)calloc(10, sizeof(unsigned int));
    d_glbSpkCntpre = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (unsigned int), glbSpkCntpre);
    glbSpkpre = (unsigned int*)calloc(100, sizeof(unsigned int));
    d_glbSpkpre = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 100 * sizeof (unsigned int), glbSpkpre);
    xpre = (scalar*)calloc(100, sizeof(scalar));
    d_xpre = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 100 * sizeof (scalar), xpre);
    shiftpre = (scalar*)calloc(10, sizeof(scalar));
    d_shiftpre = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (scalar), shiftpre);
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    inSynsyn9 = (float*)calloc(10, sizeof(float));
    d_inSynsyn9 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (float), inSynsyn9);
    inSynsyn8 = (float*)calloc(10, sizeof(float));
    d_inSynsyn8 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (float), inSynsyn8);
    inSynsyn7 = (float*)calloc(10, sizeof(float));
    d_inSynsyn7 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (float), inSynsyn7);
    inSynsyn6 = (float*)calloc(10, sizeof(float));
    d_inSynsyn6 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (float), inSynsyn6);
    inSynsyn5 = (float*)calloc(10, sizeof(float));
    d_inSynsyn5 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (float), inSynsyn5);
    inSynsyn4 = (float*)calloc(10, sizeof(float));
    d_inSynsyn4 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (float), inSynsyn4);
    inSynsyn3 = (float*)calloc(10, sizeof(float));
    d_inSynsyn3 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (float), inSynsyn3);
    inSynsyn2 = (float*)calloc(10, sizeof(float));
    d_inSynsyn2 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (float), inSynsyn2);
    inSynsyn1 = (float*)calloc(10, sizeof(float));
    d_inSynsyn1 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (float), inSynsyn1);
    inSynsyn0 = (float*)calloc(10, sizeof(float));
    d_inSynsyn0 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10 * sizeof (float), inSynsyn0);
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    wsyn0 = (scalar*)calloc(100, sizeof(scalar));
    d_wsyn0 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 100 * sizeof (scalar), wsyn0);
    wsyn1 = (scalar*)calloc(100, sizeof(scalar));
    d_wsyn1 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 100 * sizeof (scalar), wsyn1);
    wsyn2 = (scalar*)calloc(100, sizeof(scalar));
    d_wsyn2 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 100 * sizeof (scalar), wsyn2);
    wsyn3 = (scalar*)calloc(100, sizeof(scalar));
    d_wsyn3 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 100 * sizeof (scalar), wsyn3);
    wsyn4 = (scalar*)calloc(100, sizeof(scalar));
    d_wsyn4 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 100 * sizeof (scalar), wsyn4);
    wsyn5 = (scalar*)calloc(100, sizeof(scalar));
    d_wsyn5 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 100 * sizeof (scalar), wsyn5);
    wsyn6 = (scalar*)calloc(100, sizeof(scalar));
    d_wsyn6 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 100 * sizeof (scalar), wsyn6);
    wsyn7 = (scalar*)calloc(100, sizeof(scalar));
    d_wsyn7 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 100 * sizeof (scalar), wsyn7);
    wsyn8 = (scalar*)calloc(100, sizeof(scalar));
    d_wsyn8 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 100 * sizeof (scalar), wsyn8);
    wsyn9 = (scalar*)calloc(100, sizeof(scalar));
    d_wsyn9 = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 100 * sizeof (scalar), wsyn9);
    
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
    free(xpost);
    free(shiftpost);
    free(glbSpkCntpre);
    free(glbSpkpre);
    free(xpre);
    free(shiftpre);
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    free(inSynsyn9);
    free(inSynsyn8);
    free(inSynsyn7);
    free(inSynsyn6);
    free(inSynsyn5);
    free(inSynsyn4);
    free(inSynsyn3);
    free(inSynsyn2);
    free(inSynsyn1);
    free(inSynsyn0);
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    free(wsyn0);
    free(wsyn1);
    free(wsyn2);
    free(wsyn3);
    free(wsyn4);
    free(wsyn5);
    free(wsyn6);
    free(wsyn7);
    free(wsyn8);
    free(wsyn9);
    
}

void stepTime() {
    updateSynapses(t);
    spkQuePtrpre = (spkQuePtrpre + 1) % 10;
    updateNeurons(t);
    iT++;
    t = iT*DT;
}

