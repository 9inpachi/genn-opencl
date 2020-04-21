#include "definitions.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>
#include <clRNG/philox432.h>

#define DEVICE_INDEX 1

extern "C" {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    EXPORT_VAR clrngPhilox432Stream* rng;
    EXPORT_VAR cl::Buffer d_rng;

    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    EXPORT_VAR cl::Buffer d_glbSpkCntExc;
    EXPORT_VAR cl::Buffer d_glbSpkExc;
    EXPORT_VAR cl::Buffer d_VExc;
    EXPORT_VAR cl::Buffer d_UExc;
    // current source variables
    EXPORT_VAR cl::Buffer d_glbSpkCntInh;
    EXPORT_VAR cl::Buffer d_glbSpkInh;
    EXPORT_VAR cl::Buffer d_VInh;
    EXPORT_VAR cl::Buffer d_UInh;
    // current source variables

    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    EXPORT_VAR cl::Buffer d_inSynInh_Exc;
    EXPORT_VAR cl::Buffer d_inSynExc_Exc;
    EXPORT_VAR cl::Buffer d_inSynInh_Inh;
    EXPORT_VAR cl::Buffer d_inSynExc_Inh;

    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    EXPORT_VAR cl::Buffer d_rowLengthExc_Exc;
    EXPORT_VAR cl::Buffer d_indExc_Exc;
    EXPORT_VAR cl::Buffer d_rowLengthExc_Inh;
    EXPORT_VAR cl::Buffer d_indExc_Inh;
    EXPORT_VAR cl::Buffer d_rowLengthInh_Exc;
    EXPORT_VAR cl::Buffer d_indInh_Exc;
    EXPORT_VAR cl::Buffer d_rowLengthInh_Inh;
    EXPORT_VAR cl::Buffer d_indInh_Inh;

    // OpenCL variables
    EXPORT_VAR cl::Context clContext;
    EXPORT_VAR cl::Device clDevice;
    EXPORT_VAR cl::Program initProgram;
    EXPORT_VAR cl::Program updateNeuronsProgram;
    EXPORT_VAR cl::Program updateSynapsesProgram;
    EXPORT_VAR cl::CommandQueue commandQueue;

    // OpenCL kernels
    EXPORT_VAR cl::Kernel initializeKernel;
    EXPORT_VAR cl::Kernel preNeuronResetKernel;
    EXPORT_VAR cl::Kernel updateNeuronsKernel;
    EXPORT_VAR cl::Kernel updatePresynapticKernel;
    EXPORT_FUNC void initInitializationKernels();
    EXPORT_FUNC void initUpdateNeuronsKernels();
    EXPORT_FUNC void initUpdateSynapsesKernels();
    // OpenCL kernels sources
    EXPORT_VAR const char* updateNeuronsKernelSource;
    EXPORT_VAR const char* initKernelSource;
    EXPORT_VAR const char* updateSynapsesKernelSource;

}

// Declaration of OpenCL functions
namespace opencl {

    void setUpContext(cl::Context& context, cl::Device& device, const int deviceIndex);
    void createProgram(const char* kernelSource, cl::Program& program, cl::Context& context);
    std::string getCLError(cl_int errorCode);

}