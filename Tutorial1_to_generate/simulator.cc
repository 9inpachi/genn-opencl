#include "definitions.h"
#include <fstream>

int main() {
    allocateMem();
    initialize();

    aNeurons[0] = 0.02f;    bNeurons[0] = 0.2f;     cNeurons[0] = -65.0f;   dNeurons[0] = 8.0f;
    aNeurons[1] = 0.1f;     bNeurons[1] = 0.2f;     cNeurons[1] = -65.0f;   dNeurons[1] = 2.0f;
    aNeurons[2] = 0.02f;    bNeurons[2] = 0.2f;     cNeurons[2] = -50.0f;   dNeurons[2] = 2.0f;
    aNeurons[3] = 0.02f;    bNeurons[3] = 0.2f;     cNeurons[3] = -55.0f;   dNeurons[3] = 4.0f;
    aNeurons[4] = 0.02f;    bNeurons[4] = 0.25f;    cNeurons[4] = -65.0f;   dNeurons[4] = 0.05f; // For TC
    aNeurons[5] = 0.1f;     bNeurons[5] = 0.26f;    cNeurons[5] = -65.0f;   dNeurons[5] = 2.0f; // For RZ
    aNeurons[6] = 0.02f;    bNeurons[6] = 0.25f;    cNeurons[6] = -65.0f;   dNeurons[6] = 2.0f; // For LTS


    initializeSparse();

    std::ofstream stream("spikes.csv");
    while (t < 500.0f) {
        stepTime();
        pullCurrentVNeuronsFromDevice();

        scalar* currVNeurons = getCurrentVNeurons();

        stream << t << "," << currVNeurons[0] << "," << currVNeurons[1] << "," << currVNeurons[2] << "," << currVNeurons[3] << "," << currVNeurons[4] << "," << currVNeurons[5] << "," << currVNeurons[6] << std::endl;
    }

    return EXIT_SUCCESS;
}