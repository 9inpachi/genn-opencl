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
EXPORT_VAR cl::Buffer d_glbSpkCntE;
EXPORT_VAR cl::Buffer d_glbSpkE;
EXPORT_VAR cl::Buffer d_sTE;
EXPORT_VAR cl::Buffer d_VE;
EXPORT_VAR cl::Buffer d_RefracTimeE;
EXPORT_VAR cl::Buffer d_glbSpkCntI;
EXPORT_VAR cl::Buffer d_glbSpkI;
EXPORT_VAR cl::Buffer d_sTI;
EXPORT_VAR cl::Buffer d_VI;
EXPORT_VAR cl::Buffer d_RefracTimeI;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR cl::Buffer d_inSynIE;
EXPORT_VAR cl::Buffer d_inSynEE;
EXPORT_VAR cl::Buffer d_inSynII;
EXPORT_VAR cl::Buffer d_inSynEI;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
EXPORT_VAR cl::Buffer d_rowLengthEE;
EXPORT_VAR cl::Buffer d_indEE;
EXPORT_VAR cl::Buffer d_rowLengthEI;
EXPORT_VAR cl::Buffer d_indEI;
EXPORT_VAR cl::Buffer d_rowLengthIE;
EXPORT_VAR cl::Buffer d_indIE;
EXPORT_VAR cl::Buffer d_colLengthIE;
EXPORT_VAR cl::Buffer d_remapIE;
EXPORT_VAR cl::Buffer d_rowLengthII;
EXPORT_VAR cl::Buffer d_indII;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR cl::Buffer d_gIE;

}  // extern "C"
