#include "decode_matrix_globalg_ragged_pre_CODE/definitions.h"
#include <cmath>
#include <iostream>

int main() {
    std::cout << "TEST RUNNING";
    allocateMem();
    initialize();

    initializeSparse();

    for (int i = 0; i < (int)(10.0f / DT); i++) {
        // What value should neurons be representing this time step?
        const unsigned int in_value = (i / 10) + 1;

        // Input spike representing value
        // **NOTE** neurons start from zero
        glbSpkCntPre[0] = 1;
        glbSpkPre[0] = (in_value - 1);

        // Push spikes to device
        pushPreSpikesToDevice();

        // Step GeNN
        stepTime();
        copyStateFromDevice();

        // Loop through output neurons
        unsigned int out_value = 0;
        for (unsigned int j = 0; j < 4; j++) {
            // If this neuron is representing 1 add value it represents to output
            if (std::fabs(xPost[j] - 1.0f) < 1E-5) {
                out_value += (1 << j);
            }
        }

        // If input value isn't correctly decoded, return false
        if (out_value != in_value) {
            std::cout << "TEST FAILED";
            return 1;
        }
    }
    std::cout << "TEST PASSED";
}