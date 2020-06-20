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
#define spikeCount_Pop glbSpkCntPop[0]
#define spike_Pop glbSpkPop
#define glbSpkShiftPop 0

EXPORT_VAR unsigned int* glbSpkCntPop;
EXPORT_VAR unsigned int* glbSpkPop;
EXPORT_VAR scalar* constantPop;
EXPORT_VAR scalar* uniformPop;
EXPORT_VAR scalar* normalPop;
EXPORT_VAR scalar* exponentialPop;
EXPORT_VAR scalar* gammaPop;
// current source variables
EXPORT_VAR scalar* constantCurrSource;
EXPORT_VAR scalar* uniformCurrSource;
EXPORT_VAR scalar* normalCurrSource;
EXPORT_VAR scalar* exponentialCurrSource;
EXPORT_VAR scalar* gammaCurrSource;
#define spikeCount_SpikeSource glbSpkCntSpikeSource[0]
#define spike_SpikeSource glbSpkSpikeSource
#define glbSpkShiftSpikeSource 0

EXPORT_VAR unsigned int* glbSpkCntSpikeSource;
EXPORT_VAR unsigned int* glbSpkSpikeSource;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynSparse;
EXPORT_VAR scalar* pconstantSparse;
EXPORT_VAR scalar* puniformSparse;
EXPORT_VAR scalar* pnormalSparse;
EXPORT_VAR scalar* pexponentialSparse;
EXPORT_VAR scalar* pgammaSparse;
EXPORT_VAR float* inSynDense;
EXPORT_VAR scalar* pconstantDense;
EXPORT_VAR scalar* puniformDense;
EXPORT_VAR scalar* pnormalDense;
EXPORT_VAR scalar* pexponentialDense;
EXPORT_VAR scalar* pgammaDense;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
EXPORT_VAR const unsigned int maxRowLengthSparse;
EXPORT_VAR unsigned int* rowLengthSparse;
EXPORT_VAR uint32_t* indSparse;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* constantDense;
EXPORT_VAR scalar* uniformDense;
EXPORT_VAR scalar* normalDense;
EXPORT_VAR scalar* exponentialDense;
EXPORT_VAR scalar* gammaDense;
EXPORT_VAR scalar* constantSparse;
EXPORT_VAR scalar* uniformSparse;
EXPORT_VAR scalar* normalSparse;
EXPORT_VAR scalar* exponentialSparse;
EXPORT_VAR scalar* gammaSparse;

EXPORT_FUNC void pushPopSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPopSpikesFromDevice();
EXPORT_FUNC void pushPopCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPopCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getPopCurrentSpikes();
EXPORT_FUNC unsigned int& getPopCurrentSpikeCount();
EXPORT_FUNC void pushconstantPopToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullconstantPopFromDevice();
EXPORT_FUNC void pushCurrentconstantPopToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentconstantPopFromDevice();
EXPORT_FUNC scalar* getCurrentconstantPop();
EXPORT_FUNC void pushuniformPopToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulluniformPopFromDevice();
EXPORT_FUNC void pushCurrentuniformPopToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentuniformPopFromDevice();
EXPORT_FUNC scalar* getCurrentuniformPop();
EXPORT_FUNC void pushnormalPopToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullnormalPopFromDevice();
EXPORT_FUNC void pushCurrentnormalPopToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentnormalPopFromDevice();
EXPORT_FUNC scalar* getCurrentnormalPop();
EXPORT_FUNC void pushexponentialPopToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullexponentialPopFromDevice();
EXPORT_FUNC void pushCurrentexponentialPopToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentexponentialPopFromDevice();
EXPORT_FUNC scalar* getCurrentexponentialPop();
EXPORT_FUNC void pushgammaPopToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgammaPopFromDevice();
EXPORT_FUNC void pushCurrentgammaPopToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentgammaPopFromDevice();
EXPORT_FUNC scalar* getCurrentgammaPop();
EXPORT_FUNC void pushPopStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPopStateFromDevice();
EXPORT_FUNC void pushconstantCurrSourceToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullconstantCurrSourceFromDevice();
EXPORT_FUNC void pushuniformCurrSourceToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulluniformCurrSourceFromDevice();
EXPORT_FUNC void pushnormalCurrSourceToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullnormalCurrSourceFromDevice();
EXPORT_FUNC void pushexponentialCurrSourceToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullexponentialCurrSourceFromDevice();
EXPORT_FUNC void pushgammaCurrSourceToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgammaCurrSourceFromDevice();
EXPORT_FUNC void pushCurrSourceStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrSourceStateFromDevice();
EXPORT_FUNC void pushSpikeSourceSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullSpikeSourceSpikesFromDevice();
EXPORT_FUNC void pushSpikeSourceCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullSpikeSourceCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getSpikeSourceCurrentSpikes();
EXPORT_FUNC unsigned int& getSpikeSourceCurrentSpikeCount();
EXPORT_FUNC void pushSpikeSourceStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullSpikeSourceStateFromDevice();
EXPORT_FUNC void pushSparseConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullSparseConnectivityFromDevice();
EXPORT_FUNC void pushconstantDenseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullconstantDenseFromDevice();
EXPORT_FUNC void pushuniformDenseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulluniformDenseFromDevice();
EXPORT_FUNC void pushnormalDenseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullnormalDenseFromDevice();
EXPORT_FUNC void pushexponentialDenseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullexponentialDenseFromDevice();
EXPORT_FUNC void pushgammaDenseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgammaDenseFromDevice();
EXPORT_FUNC void pushinSynDenseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynDenseFromDevice();
EXPORT_FUNC void pushpconstantDenseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpconstantDenseFromDevice();
EXPORT_FUNC void pushpuniformDenseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpuniformDenseFromDevice();
EXPORT_FUNC void pushpnormalDenseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpnormalDenseFromDevice();
EXPORT_FUNC void pushpexponentialDenseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpexponentialDenseFromDevice();
EXPORT_FUNC void pushpgammaDenseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpgammaDenseFromDevice();
EXPORT_FUNC void pushDenseStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullDenseStateFromDevice();
EXPORT_FUNC void pushconstantSparseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullconstantSparseFromDevice();
EXPORT_FUNC void pushuniformSparseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulluniformSparseFromDevice();
EXPORT_FUNC void pushnormalSparseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullnormalSparseFromDevice();
EXPORT_FUNC void pushexponentialSparseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullexponentialSparseFromDevice();
EXPORT_FUNC void pushgammaSparseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgammaSparseFromDevice();
EXPORT_FUNC void pushinSynSparseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynSparseFromDevice();
EXPORT_FUNC void pushpconstantSparseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpconstantSparseFromDevice();
EXPORT_FUNC void pushpuniformSparseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpuniformSparseFromDevice();
EXPORT_FUNC void pushpnormalSparseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpnormalSparseFromDevice();
EXPORT_FUNC void pushpexponentialSparseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpexponentialSparseFromDevice();
EXPORT_FUNC void pushpgammaSparseToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpgammaSparseFromDevice();
EXPORT_FUNC void pushSparseStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullSparseStateFromDevice();
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
