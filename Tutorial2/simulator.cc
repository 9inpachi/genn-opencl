#include "definitions.h"
#include <fstream>
#include <iostream>

int main() {
	allocateMem();
	std::cout << "Initializing" << std::endl;
	initialize();
	initializeSparse();

	std::cout << "Simulating" << std::endl;
	std::ofstream file("spikes.csv");

	while (t < 2000.0f) {
		stepTime();
		pullExcCurrentSpikesFromDevice();
		pullInhCurrentSpikesFromDevice();

		unsigned int* currentExcSpikes = getExcCurrentSpikes();
		unsigned int* currentInhSpikes = getInhCurrentSpikes();

		for (unsigned int i = 0; i < getExcCurrentSpikeCount(); i++) {
			file << t << "," << currentExcSpikes[i] << std::endl;
		}

		for (unsigned int i = 0; i < getInhCurrentSpikeCount(); i++) {
			file << t << "," << 8000 + currentInhSpikes[i] << std::endl;
		}
	}

	return EXIT_SUCCESS;
}