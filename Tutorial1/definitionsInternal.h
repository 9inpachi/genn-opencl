#include "definitions.h"

extern "C" {

	// Buffers
	EXPORT_VAR cl::Buffer b_glbSpkCntNeurons;
	EXPORT_VAR cl::Buffer b_glbSpkNeurons;
	EXPORT_VAR cl::Buffer b_VNeurons;
	EXPORT_VAR cl::Buffer b_UNeurons;
	EXPORT_VAR cl::Buffer b_aNeurons;
	EXPORT_VAR cl::Buffer b_bNeurons;
	EXPORT_VAR cl::Buffer b_cNeurons;
	EXPORT_VAR cl::Buffer b_dNeurons;

	// OpenCL variables
	EXPORT_VAR cl::Context clContext;
	EXPORT_VAR cl::Device clDevice;
	EXPORT_VAR cl::Program initProgram;
	EXPORT_VAR cl::Program unProgram;
	
}