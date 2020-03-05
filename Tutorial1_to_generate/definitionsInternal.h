#include "definitions.h"

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
	
}