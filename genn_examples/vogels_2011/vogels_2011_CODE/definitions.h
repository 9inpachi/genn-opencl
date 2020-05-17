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
#define spikeCount_E glbSpkCntE[0]
#define spike_E glbSpkE
#define glbSpkShiftE 0

EXPORT_VAR unsigned int* glbSpkCntE;
EXPORT_VAR unsigned int* glbSpkE;
EXPORT_VAR float* sTE;
EXPORT_VAR scalar* VE;
EXPORT_VAR scalar* RefracTimeE;
#define spikeCount_I glbSpkCntI[0]
#define spike_I glbSpkI
#define glbSpkShiftI 0

EXPORT_VAR unsigned int* glbSpkCntI;
EXPORT_VAR unsigned int* glbSpkI;
EXPORT_VAR float* sTI;
EXPORT_VAR scalar* VI;
EXPORT_VAR scalar* RefracTimeI;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynIE;
EXPORT_VAR float* inSynEE;
EXPORT_VAR float* inSynII;
EXPORT_VAR float* inSynEI;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
EXPORT_VAR const unsigned int maxRowLengthEE;
EXPORT_VAR unsigned int* rowLengthEE;
EXPORT_VAR uint32_t* indEE;
EXPORT_VAR const unsigned int maxRowLengthEI;
EXPORT_VAR unsigned int* rowLengthEI;
EXPORT_VAR uint32_t* indEI;
EXPORT_VAR const unsigned int maxRowLengthIE;
EXPORT_VAR unsigned int* rowLengthIE;
EXPORT_VAR uint32_t* indIE;
EXPORT_VAR unsigned int* colLengthIE;
EXPORT_VAR unsigned int* remapIE;
EXPORT_VAR const unsigned int maxRowLengthII;
EXPORT_VAR unsigned int* rowLengthII;
EXPORT_VAR uint32_t* indII;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* gIE;

EXPORT_FUNC void pushESpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullESpikesFromDevice();
EXPORT_FUNC void pushECurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullECurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getECurrentSpikes();
EXPORT_FUNC unsigned int& getECurrentSpikeCount();
EXPORT_FUNC void pushESpikeTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullESpikeTimesFromDevice();
EXPORT_FUNC void pushEStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEStateFromDevice();
EXPORT_FUNC void pushISpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullISpikesFromDevice();
EXPORT_FUNC void pushICurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullICurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getICurrentSpikes();
EXPORT_FUNC unsigned int& getICurrentSpikeCount();
EXPORT_FUNC void pushIStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIStateFromDevice();
EXPORT_FUNC void pushIEConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIEConnectivityFromDevice();
EXPORT_FUNC void pushEEStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEEStateFromDevice();
EXPORT_FUNC void pushEIStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEIStateFromDevice();
EXPORT_FUNC void pushgIEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgIEFromDevice();
EXPORT_FUNC void pushIEStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIEStateFromDevice();
EXPORT_FUNC void pushIIStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIIStateFromDevice();
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
