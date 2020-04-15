#pragma once
#include <cassert>
#include <fstream>

// Standard C includes
#include <cstdint>

#define EXPORT_VAR extern
#define EXPORT_FUNC

typedef float scalar;

#define DT 0.100000f
#define NSIZE 7

typedef float scalar;

extern "C" {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    EXPORT_VAR unsigned long long iT;
    EXPORT_VAR float t;

    // ------------------------------------------------------------------------
    // remote neuron groups
    // ------------------------------------------------------------------------

    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    #define spikeCount_Exc glbSpkCntExc[0]
    #define spike_Exc glbSpkExc
    #define glbSpkShiftExc 0

    EXPORT_VAR unsigned int* glbSpkCntExc;
    EXPORT_VAR unsigned int* glbSpkExc;
    EXPORT_VAR scalar* VExc;
    EXPORT_VAR scalar* UExc;
    // current source variables
    #define spikeCount_Inh glbSpkCntInh[0]
    #define spike_Inh glbSpkInh
    #define glbSpkShiftInh 0

    EXPORT_VAR unsigned int* glbSpkCntInh;
    EXPORT_VAR unsigned int* glbSpkInh;
    EXPORT_VAR scalar* VInh;
    EXPORT_VAR scalar* UInh;
    // current source variables

    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    EXPORT_VAR float* inSynInh_Exc;
    EXPORT_VAR float* inSynExc_Exc;
    EXPORT_VAR float* inSynInh_Inh;
    EXPORT_VAR float* inSynExc_Inh;

    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    EXPORT_VAR const unsigned int maxRowLengthExc_Exc;
    EXPORT_VAR unsigned int* rowLengthExc_Exc;
    EXPORT_VAR uint32_t* indExc_Exc;
    EXPORT_VAR const unsigned int maxRowLengthExc_Inh;
    EXPORT_VAR unsigned int* rowLengthExc_Inh;
    EXPORT_VAR uint32_t* indExc_Inh;
    EXPORT_VAR const unsigned int maxRowLengthInh_Exc;
    EXPORT_VAR unsigned int* rowLengthInh_Exc;
    EXPORT_VAR uint32_t* indInh_Exc;
    EXPORT_VAR const unsigned int maxRowLengthInh_Inh;
    EXPORT_VAR unsigned int* rowLengthInh_Inh;
    EXPORT_VAR uint32_t* indInh_Inh;

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

}
