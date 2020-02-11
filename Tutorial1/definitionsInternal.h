#include "definitions.h"

extern "C" {

	// Buffers
	// initKernel buffers
	EXPORT_VAR cl::Buffer b_init_glbSpkCntNeurons;
	EXPORT_VAR cl::Buffer b_init_glbSpkNeurons;
	EXPORT_VAR cl::Buffer b_init_VNeurons;
	EXPORT_VAR cl::Buffer b_init_UNeurons;
	// updateNeuronsKernels buffers
	EXPORT_VAR cl::Buffer b_glbSpkCntNeurons;
	EXPORT_VAR cl::Buffer b_glbSpkNeurons;
	EXPORT_VAR cl::Buffer b_VNeurons;
	EXPORT_VAR cl::Buffer b_UNeurons;
	EXPORT_VAR cl::Buffer b_aNeurons;
	EXPORT_VAR cl::Buffer b_bNeurons;
	EXPORT_VAR cl::Buffer b_cNeurons;
	EXPORT_VAR cl::Buffer b_dNeurons;

	// OpenCL variables
	// initKernel OpenCL variables
	EXPORT_VAR cl::Program initProgram;
	EXPORT_VAR cl::Context initContext;
	EXPORT_VAR cl::Device initDevice;
	// updateNeuronsKernels OpenCL variables
	EXPORT_VAR cl::Program unProgram;
	EXPORT_VAR cl::Context unContext;
	EXPORT_VAR cl::Device unDevice;
	
}