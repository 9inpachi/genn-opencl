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
#define DT 0.100000f
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
#define spikeCount_post glbSpkCntpost[0]
#define spike_post glbSpkpost
#define glbSpkShiftpost 0

EXPORT_VAR unsigned int* glbSpkCntpost;
EXPORT_VAR unsigned int* glbSpkpost;
EXPORT_VAR scalar* xpost;
EXPORT_VAR scalar* shiftpost;
#define spikeCount_pre glbSpkCntpre[spkQuePtrpre]
#define spike_pre (glbSpkpre + (spkQuePtrpre * 10))
#define glbSpkShiftpre spkQuePtrpre*10

EXPORT_VAR unsigned int* glbSpkCntpre;
EXPORT_VAR unsigned int* glbSpkpre;
EXPORT_VAR unsigned int spkQuePtrpre;
EXPORT_VAR scalar* xpre;
EXPORT_VAR scalar* shiftpre;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynsyn9;
EXPORT_VAR float* inSynsyn8;
EXPORT_VAR float* inSynsyn7;
EXPORT_VAR float* inSynsyn6;
EXPORT_VAR float* inSynsyn5;
EXPORT_VAR float* inSynsyn4;
EXPORT_VAR float* inSynsyn3;
EXPORT_VAR float* inSynsyn2;
EXPORT_VAR float* inSynsyn1;
EXPORT_VAR float* inSynsyn0;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* wsyn0;
EXPORT_VAR scalar* wsyn1;
EXPORT_VAR scalar* wsyn2;
EXPORT_VAR scalar* wsyn3;
EXPORT_VAR scalar* wsyn4;
EXPORT_VAR scalar* wsyn5;
EXPORT_VAR scalar* wsyn6;
EXPORT_VAR scalar* wsyn7;
EXPORT_VAR scalar* wsyn8;
EXPORT_VAR scalar* wsyn9;

EXPORT_FUNC void pushpostSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpostSpikesFromDevice();
EXPORT_FUNC void pushpostCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpostCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getpostCurrentSpikes();
EXPORT_FUNC unsigned int& getpostCurrentSpikeCount();
EXPORT_FUNC void pushxpostToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullxpostFromDevice();
EXPORT_FUNC void pushCurrentxpostToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentxpostFromDevice();
EXPORT_FUNC scalar* getCurrentxpost();
EXPORT_FUNC void pushshiftpostToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullshiftpostFromDevice();
EXPORT_FUNC void pushCurrentshiftpostToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentshiftpostFromDevice();
EXPORT_FUNC scalar* getCurrentshiftpost();
EXPORT_FUNC void pushpostStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpostStateFromDevice();
EXPORT_FUNC void pushpreSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpreSpikesFromDevice();
EXPORT_FUNC void pushpreCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpreCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getpreCurrentSpikes();
EXPORT_FUNC unsigned int& getpreCurrentSpikeCount();
EXPORT_FUNC void pushxpreToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullxpreFromDevice();
EXPORT_FUNC void pushCurrentxpreToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentxpreFromDevice();
EXPORT_FUNC scalar* getCurrentxpre();
EXPORT_FUNC void pushshiftpreToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullshiftpreFromDevice();
EXPORT_FUNC void pushCurrentshiftpreToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentshiftpreFromDevice();
EXPORT_FUNC scalar* getCurrentshiftpre();
EXPORT_FUNC void pushpreStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpreStateFromDevice();
EXPORT_FUNC void pushwsyn0ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwsyn0FromDevice();
EXPORT_FUNC void pushinSynsyn0ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn0FromDevice();
EXPORT_FUNC void pushsyn0StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn0StateFromDevice();
EXPORT_FUNC void pushwsyn1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwsyn1FromDevice();
EXPORT_FUNC void pushinSynsyn1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn1FromDevice();
EXPORT_FUNC void pushsyn1StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn1StateFromDevice();
EXPORT_FUNC void pushwsyn2ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwsyn2FromDevice();
EXPORT_FUNC void pushinSynsyn2ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn2FromDevice();
EXPORT_FUNC void pushsyn2StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn2StateFromDevice();
EXPORT_FUNC void pushwsyn3ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwsyn3FromDevice();
EXPORT_FUNC void pushinSynsyn3ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn3FromDevice();
EXPORT_FUNC void pushsyn3StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn3StateFromDevice();
EXPORT_FUNC void pushwsyn4ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwsyn4FromDevice();
EXPORT_FUNC void pushinSynsyn4ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn4FromDevice();
EXPORT_FUNC void pushsyn4StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn4StateFromDevice();
EXPORT_FUNC void pushwsyn5ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwsyn5FromDevice();
EXPORT_FUNC void pushinSynsyn5ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn5FromDevice();
EXPORT_FUNC void pushsyn5StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn5StateFromDevice();
EXPORT_FUNC void pushwsyn6ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwsyn6FromDevice();
EXPORT_FUNC void pushinSynsyn6ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn6FromDevice();
EXPORT_FUNC void pushsyn6StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn6StateFromDevice();
EXPORT_FUNC void pushwsyn7ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwsyn7FromDevice();
EXPORT_FUNC void pushinSynsyn7ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn7FromDevice();
EXPORT_FUNC void pushsyn7StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn7StateFromDevice();
EXPORT_FUNC void pushwsyn8ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwsyn8FromDevice();
EXPORT_FUNC void pushinSynsyn8ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn8FromDevice();
EXPORT_FUNC void pushsyn8StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn8StateFromDevice();
EXPORT_FUNC void pushwsyn9ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwsyn9FromDevice();
EXPORT_FUNC void pushinSynsyn9ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn9FromDevice();
EXPORT_FUNC void pushsyn9StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn9StateFromDevice();
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
