#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>
#include <cassert>
#include <fstream>

#define EXPORT_VAR extern
#define EXPORT_FUNC

typedef float scalar;

#define DT 0.100000f
#define NSIZE 7
#define DEVICE_INDEX 1

extern "C" {

	EXPORT_VAR unsigned int* glbSpkCntNeurons;
	EXPORT_VAR unsigned int* glbSpkNeurons;
	EXPORT_VAR scalar* VNeurons;
	EXPORT_VAR scalar* UNeurons;
	EXPORT_VAR scalar* aNeurons;
	EXPORT_VAR scalar* bNeurons;
	EXPORT_VAR scalar* cNeurons;
	EXPORT_VAR scalar* dNeurons;
	EXPORT_VAR unsigned long long iT;
	EXPORT_VAR float t;

	EXPORT_FUNC void updateNeurons(float t);
	EXPORT_FUNC void allocateMem();
	EXPORT_FUNC void initialize();
	EXPORT_FUNC void initializeSparse();
	EXPORT_FUNC void stepTime();
	EXPORT_FUNC scalar* getCurrentVNeurons();
	EXPORT_FUNC void pullCurrentVNeuronsFromDevice();

}
