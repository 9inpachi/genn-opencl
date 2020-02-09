#include "definitions.h"

extern "C" {
	unsigned int* dd_glbSpkCntNeurons;
	unsigned int* dd_glbSpkNeurons;
	scalar* dd_VNeurons;
	scalar* dd_UNeurons;
	scalar* dd_aNeurons;
	scalar* dd_bNeurons;
	scalar* dd_cNeurons;
	scalar* dd_dNeurons;
	unsigned long long iT;
	float t;
}

void allocateMem() {
	dd_glbSpkCntNeurons = (unsigned int*) malloc(1 * sizeof(unsigned int));
	dd_glbSpkNeurons = (unsigned int*)malloc(NSIZE * sizeof(unsigned int));
	dd_VNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	dd_UNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	dd_aNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	dd_bNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	dd_cNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
	dd_dNeurons = (scalar*)malloc(NSIZE * sizeof(scalar));
}

void stepTime() {
	updateNeurons(t);
	iT++;
	t = iT * DT;
}

scalar* getCurrentVNeurons() {
	return dd_VNeurons;
}