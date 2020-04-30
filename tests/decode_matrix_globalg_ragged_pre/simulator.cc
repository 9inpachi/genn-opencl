#include "decode_matrix_globalg_ragged_pre_CODE/definitions.h"
#include <cmath>
#include <iostream>

int main() {
    std::cout << "TEST RUNNING" << std::endl;

    allocateMem();

    // Loop through presynaptic neurons
    for (unsigned int i = 0; i < 10; i++)
    {
        // Initially zero row length
        rowLengthSyn[i] = 0;
        for (unsigned int j = 0; j < 4; j++)
        {
            // Get value this post synaptic neuron represents
            const unsigned int j_value = (1 << j);

            // If this postsynaptic neuron should be connected, add index
            if (((i + 1) & j_value) != 0)
            {
                const unsigned int idx = (i * 4) + rowLengthSyn[i]++;
                indSyn[idx] = j;
            }
        }
    }

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
            std::cout << xPost[j] << std::endl;
            // If this neuron is representing 1 add value it represents to output
            if (std::fabs(xPost[j] - 1.0f) < 1E-5) {
                out_value += (1 << j);
            }
        }

        std::cout << "LOOP: " << i << " - out = " << out_value << " - in = " << in_value << std::endl;

        // If input value isn't correctly decoded, return false
        if (out_value != in_value) {
            std::cout << "TEST FAILED" << std::endl;
            return 1;
        }
    }
    std::cout << "TEST PASSED" << std::endl;
}