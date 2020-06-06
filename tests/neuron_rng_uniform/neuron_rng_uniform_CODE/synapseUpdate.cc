#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateSynapsesProgramSrc = R"(typedef float scalar;
typedef unsigned char uint8_t;

#define fmodf fmod
// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

// ------------------------------------------------------------------------
// support code
// ------------------------------------------------------------------------
#define SUPPORT_CODE_FUNC
// support code for neuron groups

// support code for synapse groups

)";

// Initialize the synapse update kernel(s)
void updateSynapsesProgramKernels() {
}

void updateSynapses(float t) {
}
