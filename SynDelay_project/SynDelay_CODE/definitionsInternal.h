#pragma once
#include "definitions.h"

#pragma warning(disable: 4297)
// OpenCL includes
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>

#define DEVICE_INDEX 1

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
    char* clGetErrorString(cl_int error);
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
EXPORT_VAR cl::Buffer d_glbSpkCntInput;
EXPORT_VAR cl::Buffer d_glbSpkInput;
EXPORT_VAR cl::Buffer d_spkQuePtrInput;
EXPORT_VAR cl::Buffer d_VInput;
EXPORT_VAR cl::Buffer d_UInput;
// current source variables
EXPORT_VAR cl::Buffer d_glbSpkCntInter;
EXPORT_VAR cl::Buffer d_glbSpkInter;
EXPORT_VAR cl::Buffer d_VInter;
EXPORT_VAR cl::Buffer d_UInter;
EXPORT_VAR cl::Buffer d_glbSpkCntOutput;
EXPORT_VAR cl::Buffer d_glbSpkOutput;
EXPORT_VAR cl::Buffer d_VOutput;
EXPORT_VAR cl::Buffer d_UOutput;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR cl::Buffer d_inSynInputInter;
EXPORT_VAR cl::Buffer d_inSynInterOutput;
EXPORT_VAR cl::Buffer d_inSynInputOutput;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

}  // extern "C"
