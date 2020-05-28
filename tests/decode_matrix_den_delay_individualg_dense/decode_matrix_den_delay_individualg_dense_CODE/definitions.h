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
#define spikeCount_Post glbSpkCntPost[0]
#define spike_Post glbSpkPost
#define glbSpkShiftPost 0

EXPORT_VAR unsigned int* glbSpkCntPost;
EXPORT_VAR unsigned int* glbSpkPost;
EXPORT_VAR scalar* xPost;
#define spikeCount_Pre glbSpkCntPre[0]
#define spike_Pre glbSpkPre
#define glbSpkShiftPre 0

EXPORT_VAR unsigned int* glbSpkCntPre;
EXPORT_VAR unsigned int* glbSpkPre;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynSyn;
EXPORT_VAR float* denDelaySyn;
EXPORT_VAR unsigned int denDelayPtrSyn;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* gSyn;
EXPORT_VAR uint8_t* dSyn;

EXPORT_FUNC void pushPostSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPostSpikesFromDevice();
EXPORT_FUNC void pushPostCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPostCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getPostCurrentSpikes();
EXPORT_FUNC unsigned int& getPostCurrentSpikeCount();
EXPORT_FUNC void pushxPostToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullxPostFromDevice();
EXPORT_FUNC void pushCurrentxPostToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentxPostFromDevice();
EXPORT_FUNC scalar* getCurrentxPost();
EXPORT_FUNC void pushPostStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPostStateFromDevice();
EXPORT_FUNC void pushPreSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPreSpikesFromDevice();
EXPORT_FUNC void pushPreCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPreCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getPreCurrentSpikes();
EXPORT_FUNC unsigned int& getPreCurrentSpikeCount();
EXPORT_FUNC void pushPreStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPreStateFromDevice();
EXPORT_FUNC void pushgSynToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgSynFromDevice();
EXPORT_FUNC void pushdSynToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldSynFromDevice();
EXPORT_FUNC void pushinSynSynToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynSynFromDevice();
EXPORT_FUNC void pushSynStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullSynStateFromDevice();
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
