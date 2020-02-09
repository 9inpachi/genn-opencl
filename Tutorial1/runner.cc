#include "definitions.h"

void stepTime() {
	updateNeurons(t);
	iT++;
	t = iT * DT;
}

std::vector<scalar> getCurrentVNeurons() {
	return dd_VNeurons;
}