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
#define spikeCount_Input glbSpkCntInput[spkQuePtrInput]
#define spike_Input (glbSpkInput + (spkQuePtrInput * 500))
#define glbSpkShiftInput spkQuePtrInput*500

EXPORT_VAR unsigned int* glbSpkCntInput;
EXPORT_VAR unsigned int* glbSpkInput;
EXPORT_VAR unsigned int spkQuePtrInput;
EXPORT_VAR scalar* VInput;
EXPORT_VAR scalar* UInput;
// current source variables
#define spikeCount_Inter glbSpkCntInter[0]
#define spike_Inter glbSpkInter
#define glbSpkShiftInter 0

EXPORT_VAR unsigned int* glbSpkCntInter;
EXPORT_VAR unsigned int* glbSpkInter;
EXPORT_VAR scalar* VInter;
EXPORT_VAR scalar* UInter;
#define spikeCount_Output glbSpkCntOutput[0]
#define spike_Output glbSpkOutput
#define glbSpkShiftOutput 0

EXPORT_VAR unsigned int* glbSpkCntOutput;
EXPORT_VAR unsigned int* glbSpkOutput;
EXPORT_VAR scalar* VOutput;
EXPORT_VAR scalar* UOutput;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynInputInter;
EXPORT_VAR float* inSynInterOutput;
EXPORT_VAR float* inSynInputOutput;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

EXPORT_FUNC void pushInputSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInputSpikesFromDevice();
EXPORT_FUNC void pushInputCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInputCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getInputCurrentSpikes();
EXPORT_FUNC unsigned int& getInputCurrentSpikeCount();
EXPORT_FUNC void pushVInputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVInputFromDevice();
EXPORT_FUNC void pushCurrentVInputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVInputFromDevice();
EXPORT_FUNC scalar* getCurrentVInput();
EXPORT_FUNC void pushUInputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullUInputFromDevice();
EXPORT_FUNC void pushCurrentUInputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentUInputFromDevice();
EXPORT_FUNC scalar* getCurrentUInput();
EXPORT_FUNC void pushInputStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInputStateFromDevice();
EXPORT_FUNC void pushInputCurrentSourceStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInputCurrentSourceStateFromDevice();
EXPORT_FUNC void pushInterSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInterSpikesFromDevice();
EXPORT_FUNC void pushInterCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInterCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getInterCurrentSpikes();
EXPORT_FUNC unsigned int& getInterCurrentSpikeCount();
EXPORT_FUNC void pushVInterToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVInterFromDevice();
EXPORT_FUNC void pushCurrentVInterToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVInterFromDevice();
EXPORT_FUNC scalar* getCurrentVInter();
EXPORT_FUNC void pushUInterToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullUInterFromDevice();
EXPORT_FUNC void pushCurrentUInterToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentUInterFromDevice();
EXPORT_FUNC scalar* getCurrentUInter();
EXPORT_FUNC void pushInterStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInterStateFromDevice();
EXPORT_FUNC void pushOutputSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullOutputSpikesFromDevice();
EXPORT_FUNC void pushOutputCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullOutputCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getOutputCurrentSpikes();
EXPORT_FUNC unsigned int& getOutputCurrentSpikeCount();
EXPORT_FUNC void pushVOutputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVOutputFromDevice();
EXPORT_FUNC void pushCurrentVOutputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVOutputFromDevice();
EXPORT_FUNC scalar* getCurrentVOutput();
EXPORT_FUNC void pushUOutputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullUOutputFromDevice();
EXPORT_FUNC void pushCurrentUOutputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentUOutputFromDevice();
EXPORT_FUNC scalar* getCurrentUOutput();
EXPORT_FUNC void pushOutputStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullOutputStateFromDevice();
EXPORT_FUNC void pushinSynInputInterToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynInputInterFromDevice();
EXPORT_FUNC void pushInputInterStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInputInterStateFromDevice();
EXPORT_FUNC void pushinSynInputOutputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynInputOutputFromDevice();
EXPORT_FUNC void pushInputOutputStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInputOutputStateFromDevice();
EXPORT_FUNC void pushinSynInterOutputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynInterOutputFromDevice();
EXPORT_FUNC void pushInterOutputStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInterOutputStateFromDevice();
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
