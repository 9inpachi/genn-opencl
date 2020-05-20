#pragma once
#ifdef BUILDING_GENERATED_CODE
#define EXPORT_VAR __declspec(dllexport) extern
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_VAR __declspec(dllimport) extern
#define EXPORT_FUNC __declspec(dllimport)
#endif
// Standard C++ includes
#include <string>
#include <stdexcept>

// Standard C includes
#include <cstdint>
#include <cassert>
#define DT 1.000000f
typedef float scalar;
#define SCALAR_MIN 1.175494351e-38f
#define SCALAR_MAX 3.402823466e+38f

#define TIME_MIN 1.175494351e-38f
#define TIME_MAX 3.402823466e+38f

// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR unsigned long long iT;
EXPORT_VAR float t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
EXPORT_VAR double neuronUpdateTime;
EXPORT_VAR double initTime;
EXPORT_VAR double presynapticUpdateTime;
EXPORT_VAR double postsynapticUpdateTime;
EXPORT_VAR double synapseDynamicsTime;
EXPORT_VAR double initSparseTime;
// ------------------------------------------------------------------------
// remote neuron groups
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
#define spikeCount_post glbSpkCntpost[spkQuePtrpost]
#define spike_post (glbSpkpost + (spkQuePtrpost * 10))
#define glbSpkShiftpost spkQuePtrpost*10

EXPORT_VAR unsigned int* glbSpkCntpost;
EXPORT_VAR unsigned int* glbSpkpost;
EXPORT_VAR unsigned int spkQuePtrpost;
#define spikeCount_pre glbSpkCntpre[0]
#define spike_pre glbSpkpre
#define glbSpkShiftpre 0

EXPORT_VAR unsigned int* glbSpkCntpre;
EXPORT_VAR unsigned int* glbSpkpre;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynsyn;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
EXPORT_VAR const unsigned int maxRowLengthsyn;
EXPORT_VAR unsigned int* rowLengthsyn;
EXPORT_VAR uint32_t* indsyn;
EXPORT_VAR unsigned int* colLengthsyn;
EXPORT_VAR unsigned int* remapsyn;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* wsyn;
EXPORT_VAR scalar* ssyn;

EXPORT_FUNC void pushpostSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpostSpikesFromDevice();
EXPORT_FUNC void pushpostCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpostCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getpostCurrentSpikes();
EXPORT_FUNC unsigned int& getpostCurrentSpikeCount();
EXPORT_FUNC void pushpostStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpostStateFromDevice();
EXPORT_FUNC void pushpreSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpreSpikesFromDevice();
EXPORT_FUNC void pushpreCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpreCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getpreCurrentSpikes();
EXPORT_FUNC unsigned int& getpreCurrentSpikeCount();
EXPORT_FUNC void pushpreStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpreStateFromDevice();
EXPORT_FUNC void pushsynConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsynConnectivityFromDevice();
EXPORT_FUNC void pushwsynToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwsynFromDevice();
EXPORT_FUNC void pushssynToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullssynFromDevice();
EXPORT_FUNC void pushinSynsynToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsynFromDevice();
EXPORT_FUNC void pushsynStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsynStateFromDevice();
// Runner functions
EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyStateFromDevice();
EXPORT_FUNC void copyCurrentSpikesFromDevice();
EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();
EXPORT_FUNC void allocateMem();
EXPORT_FUNC void freeMem();
EXPORT_FUNC void stepTime();

// Functions generated by backend
EXPORT_FUNC void updateNeurons(float t);
EXPORT_FUNC void updateSynapses(float t);
EXPORT_FUNC void initialize();
EXPORT_FUNC void initializeSparse();
}  // extern "C"
