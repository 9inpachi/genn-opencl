#pragma once
#include <cassert>
#include <fstream>

#define EXPORT_VAR extern
#define EXPORT_FUNC

typedef float scalar;

#define DT 0.100000f
#define NSIZE 7

extern "C" {

    EXPORT_VAR unsigned int* glbSpkCntNeurons;
    EXPORT_VAR unsigned int* glbSpkNeurons;
    EXPORT_VAR scalar* VNeurons;
    EXPORT_VAR scalar* UNeurons;
    EXPORT_VAR scalar* aNeurons;
    EXPORT_VAR scalar* bNeurons;
    EXPORT_VAR scalar* cNeurons;
    EXPORT_VAR scalar* dNeurons;
    EXPORT_VAR unsigned long long iT;
    EXPORT_VAR float t;

    EXPORT_FUNC void updateNeurons(float t);
    EXPORT_FUNC void initialize();
    EXPORT_FUNC void initializeSparse();
    EXPORT_FUNC void stepTime();
    EXPORT_FUNC scalar* getCurrentVNeurons();
    EXPORT_FUNC void pullCurrentVNeuronsFromDevice();
    EXPORT_FUNC void pushVNeuronsToDevice(bool uninitialisedOnly = false);
    EXPORT_FUNC void pushCurrentVNeuronsToDevice(bool uninitialisedOnly = false);
    EXPORT_FUNC void pushUNeuronsToDevice(bool uninitialisedOnly = false);
    EXPORT_FUNC void pushaNeuronsToDevice(bool uninitialisedOnly = false);
    EXPORT_FUNC void pushCurrentaNeuronsToDevice(bool uninitialisedOnly = false);
    EXPORT_FUNC void pushbNeuronsToDevice(bool uninitialisedOnly = false);
    EXPORT_FUNC void pushCurrentbNeuronsToDevice(bool uninitialisedOnly = false);
    EXPORT_FUNC void pushcNeuronsToDevice(bool uninitialisedOnly = false);
    EXPORT_FUNC void pushCurrentcNeuronsToDevice(bool uninitialisedOnly = false);
    EXPORT_FUNC void pushdNeuronsToDevice(bool uninitialisedOnly = false);
    EXPORT_FUNC void pushCurrentdNeuronsToDevice(bool uninitialisedOnly = false);
    EXPORT_FUNC void pushNeuronsStateToDevice(bool uninitialisedOnly = false);
    // Runner functions
    EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);
    EXPORT_FUNC void allocateMem();

}
