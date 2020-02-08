#pragma once
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
	EXPORT_FUNC void updateNeurons(float t);
}