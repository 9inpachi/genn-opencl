#include "definitions.h"
#include <fstream>

int main() {
	allocateMem();

	dd_aNeurons[0] = 0.02f;	dd_bNeurons[0] = 0.2f;	dd_cNeurons[0] = -65.0f;		dd_dNeurons[0] = 8.0f;
	dd_aNeurons[1] = 0.1f;	dd_bNeurons[1] = 0.2f;	dd_cNeurons[1] = -65.0f;		dd_dNeurons[1] = 2.0f;
	dd_aNeurons[2] = 0.02f;	dd_bNeurons[2] = 0.2f;	dd_cNeurons[2] = -50.0f;		dd_dNeurons[2] = 2.0f;
	dd_aNeurons[3] = 0.02f;	dd_bNeurons[3] = 0.2f;	dd_cNeurons[3] = -55.0f;		dd_dNeurons[3] = 4.0f;
	dd_aNeurons[4] = 0.02f;	dd_bNeurons[4] = 0.25f;	dd_cNeurons[4] = -65.0f;		dd_dNeurons[4] = 0.05f; // For TC
	dd_aNeurons[5] = 0.1f;	dd_bNeurons[5] = 0.26f;	dd_cNeurons[5] = -65.0f;		dd_dNeurons[5] = 2.0f; // For RZ
	dd_aNeurons[6] = 0.02f;	dd_bNeurons[6] = 0.25f;	dd_cNeurons[6] = -65.0f;		dd_dNeurons[6] = 2.0f; // For LTS

	initializeOpenCL();
	initialize();

	std::ofstream stream("spikes.csv");
	while (t < 500.0f) {
		stepTime();
		pullCurrentVNeuronsFromDevice();

		scalar* currVNeurons = getCurrentVNeurons();

		stream << t << "," << currVNeurons[0] << "," << currVNeurons[1] << "," << currVNeurons[2] << "," << currVNeurons[3] << "," << currVNeurons[4] << "," << currVNeurons[5] << "," << currVNeurons[6] << std::endl;
	}

	return EXIT_SUCCESS;
}