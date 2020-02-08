#include "definitionsInternal.h"

extern "C" {
	unsigned long long iT;
	float t;

	unsigned int* dd_glbSpkNeurons;
	unsigned int* dd_glbSpkCntNeurons;

	scalar* aNeurons;
}

void stepTime() {
	updateNeurons(t);
	iT++;
	t = iT * DT;
}