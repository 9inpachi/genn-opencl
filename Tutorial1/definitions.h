#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>
#ifdef BUILDING_GENERATED_CODE
#define EXPORT_VAR __declspec(dllexport) extern
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_VAR __declspec(dllimport) extern
#define EXPORT_FUNC __declspec(dllimport)
#endif

typedef float scalar;

#define DT 0.100000f

extern "C" {
	std::vector<unsigned int> dd_glbSpkCntNeurons(1); // __device__
	std::vector<unsigned int> dd_glbSpkNeurons(7); // __device__
	std::vector<scalar> dd_VNeurons(7); // __device__
	std::vector<scalar> dd_UNeurons(7); // __device__
	std::vector<scalar> dd_aNeurons(7); // __device__
	std::vector<scalar> dd_bNeurons(7); // __device__
	std::vector<scalar> dd_cNeurons(7); // __device__
	std::vector<scalar> dd_dNeurons(7); // __device__
	EXPORT_VAR unsigned long long iT;
	EXPORT_VAR float t;

	EXPORT_FUNC void updateNeurons(float t);
	EXPORT_FUNC void initialize();
	EXPORT_FUNC void stepTime();
	EXPORT_FUNC std::vector<scalar> getCurrentVNeurons();
}