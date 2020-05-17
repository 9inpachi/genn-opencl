#pragma once
#include "definitions.h"

#pragma warning(disable: 4297)
// OpenCL includes
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>

#define DEVICE_INDEX 0

// ------------------------------------------------------------------------
// Helper macro for error-checking OpenCL calls
#define CHECK_OPENCL_ERRORS(call) {\
    cl_int error = call;\
    if (error != CL_SUCCESS) {\
        throw std::runtime_error(__FILE__": " + std::to_string(__LINE__) + ": opencl error " + std::to_string(error) + ": " + opencl::clGetErrorString(error));\
    }\
}

// ------------------------------------------------------------------------
// OpenCL functions declaration
// ------------------------------------------------------------------------
namespace opencl {
    void setUpContext(cl::Context& context, cl::Device& device, const int deviceIndex);
    void createProgram(const char* kernelSource, cl::Program& program, cl::Context& context);
    const char* clGetErrorString(cl_int error);
}

extern "C" {
// OpenCL variables
EXPORT_VAR cl::Context clContext;
EXPORT_VAR cl::Device clDevice;
EXPORT_VAR cl::CommandQueue commandQueue;

// OpenCL programs
EXPORT_VAR cl::Program initProgram;
EXPORT_VAR cl::Program updateNeuronsProgram;
EXPORT_VAR cl::Program updateSynapsesProgram;

// OpenCL kernels
EXPORT_VAR cl::Kernel updateNeuronsKernel;
EXPORT_VAR cl::Kernel updatePresynapticKernel;
EXPORT_VAR cl::Kernel updatePostsynapticKernel;
EXPORT_VAR cl::Kernel updateSynapseDynamicsKernel;
EXPORT_VAR cl::Kernel initializeKernel;
EXPORT_VAR cl::Kernel initializeSparseKernel;
EXPORT_VAR cl::Kernel preNeuronResetKernel;
EXPORT_VAR cl::Kernel preSynapseResetKernel;
// OpenCL kernels initialization functions and kernels sources
EXPORT_FUNC void initProgramKernels();
EXPORT_VAR const char* initProgramSrc;
EXPORT_FUNC void updateNeuronsProgramKernels();
EXPORT_VAR const char* updateNeuronsProgramSrc;
EXPORT_FUNC void updateSynapsesProgramKernels();
EXPORT_VAR const char* updateSynapsesProgramSrc;
} // extern "C"

extern "C" {
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
EXPORT_VAR cl::Buffer d_glbSpkCntpost;
EXPORT_VAR cl::Buffer d_glbSpkpost;
EXPORT_VAR cl::Buffer d_xpost;
EXPORT_VAR cl::Buffer d_shiftpost;
EXPORT_VAR cl::Buffer d_glbSpkCntpre;
EXPORT_VAR cl::Buffer d_glbSpkpre;
EXPORT_VAR cl::Buffer d_spkQuePtrpre;
EXPORT_VAR cl::Buffer d_xpre;
EXPORT_VAR cl::Buffer d_shiftpre;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR cl::Buffer d_inSynsyn9;
EXPORT_VAR cl::Buffer d_inSynsyn8;
EXPORT_VAR cl::Buffer d_inSynsyn7;
EXPORT_VAR cl::Buffer d_inSynsyn6;
EXPORT_VAR cl::Buffer d_inSynsyn5;
EXPORT_VAR cl::Buffer d_inSynsyn4;
EXPORT_VAR cl::Buffer d_inSynsyn3;
EXPORT_VAR cl::Buffer d_inSynsyn2;
EXPORT_VAR cl::Buffer d_inSynsyn1;
EXPORT_VAR cl::Buffer d_inSynsyn0;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR cl::Buffer d_wsyn0;
EXPORT_VAR cl::Buffer d_wsyn1;
EXPORT_VAR cl::Buffer d_wsyn2;
EXPORT_VAR cl::Buffer d_wsyn3;
EXPORT_VAR cl::Buffer d_wsyn4;
EXPORT_VAR cl::Buffer d_wsyn5;
EXPORT_VAR cl::Buffer d_wsyn6;
EXPORT_VAR cl::Buffer d_wsyn7;
EXPORT_VAR cl::Buffer d_wsyn8;
EXPORT_VAR cl::Buffer d_wsyn9;

}  // extern "C"
