#pragma once
#include "definitions.h"

#pragma warning(disable: 4297)
// OpenCL includes
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>
#include <clRNG/lfsr113.h>
#include <clRNG/philox432.h>

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
EXPORT_VAR clrngLfsr113Stream* rng;
EXPORT_VAR cl::Buffer d_rng;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// remote neuron groups
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
EXPORT_VAR cl::Buffer d_glbSpkCntPop;
EXPORT_VAR cl::Buffer d_glbSpkPop;
EXPORT_VAR cl::Buffer d_constantPop;
EXPORT_VAR cl::Buffer d_uniformPop;
EXPORT_VAR cl::Buffer d_normalPop;
EXPORT_VAR cl::Buffer d_exponentialPop;
EXPORT_VAR cl::Buffer d_gammaPop;
// current source variables
EXPORT_VAR cl::Buffer d_constantCurrSource;
EXPORT_VAR cl::Buffer d_uniformCurrSource;
EXPORT_VAR cl::Buffer d_normalCurrSource;
EXPORT_VAR cl::Buffer d_exponentialCurrSource;
EXPORT_VAR cl::Buffer d_gammaCurrSource;
EXPORT_VAR cl::Buffer d_glbSpkCntSpikeSource;
EXPORT_VAR cl::Buffer d_glbSpkSpikeSource;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR cl::Buffer d_inSynSparse;
EXPORT_VAR cl::Buffer d_pconstantSparse;
EXPORT_VAR cl::Buffer d_puniformSparse;
EXPORT_VAR cl::Buffer d_pnormalSparse;
EXPORT_VAR cl::Buffer d_pexponentialSparse;
EXPORT_VAR cl::Buffer d_pgammaSparse;
EXPORT_VAR cl::Buffer d_inSynDense;
EXPORT_VAR cl::Buffer d_pconstantDense;
EXPORT_VAR cl::Buffer d_puniformDense;
EXPORT_VAR cl::Buffer d_pnormalDense;
EXPORT_VAR cl::Buffer d_pexponentialDense;
EXPORT_VAR cl::Buffer d_pgammaDense;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
EXPORT_VAR cl::Buffer d_rowLengthSparse;
EXPORT_VAR cl::Buffer d_indSparse;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR cl::Buffer d_constantDense;
EXPORT_VAR cl::Buffer d_uniformDense;
EXPORT_VAR cl::Buffer d_normalDense;
EXPORT_VAR cl::Buffer d_exponentialDense;
EXPORT_VAR cl::Buffer d_gammaDense;
EXPORT_VAR cl::Buffer d_constantSparse;
EXPORT_VAR cl::Buffer d_uniformSparse;
EXPORT_VAR cl::Buffer d_normalSparse;
EXPORT_VAR cl::Buffer d_exponentialSparse;
EXPORT_VAR cl::Buffer d_gammaSparse;

}  // extern "C"
