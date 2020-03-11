#include "definitions.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>

#define DEVICE_INDEX 1

extern "C" {

	// Buffers
	EXPORT_VAR cl::Buffer db_glbSpkCntNeurons;
	EXPORT_VAR cl::Buffer db_glbSpkNeurons;
	EXPORT_VAR cl::Buffer db_VNeurons;
	EXPORT_VAR cl::Buffer db_UNeurons;
	EXPORT_VAR cl::Buffer db_aNeurons;
	EXPORT_VAR cl::Buffer db_bNeurons;
	EXPORT_VAR cl::Buffer db_cNeurons;
	EXPORT_VAR cl::Buffer db_dNeurons;

	// OpenCL variables
	EXPORT_VAR cl::Context clContext;
	EXPORT_VAR cl::Device clDevice;
	EXPORT_VAR cl::Program initProgram;
	EXPORT_VAR cl::Program unProgram;
	EXPORT_VAR cl::CommandQueue commandQueue;

	// OpenCL kernels
	EXPORT_VAR cl::Kernel initKernel;
	EXPORT_VAR cl::Kernel preNeuronResetKernel;
	EXPORT_VAR cl::Kernel updateNeuronsKernel;
	EXPORT_FUNC void initUpdateNeuronsKernels();
	EXPORT_FUNC void initInitKernel();
	// OpenCL kernels sources
	EXPORT_VAR const char* updateNeuronsKernelSource;
	EXPORT_VAR const char* initKernelSource;
	
}

// Declaration of OpenCL functions
namespace opencl {

	void setUpContext(cl::Context& context, cl::Device& device, const int deviceIndex);
	void createProgram(const char* kernelSource, cl::Program& program, cl::Context& context);
	std::string getCLError(cl_int errorCode);

}