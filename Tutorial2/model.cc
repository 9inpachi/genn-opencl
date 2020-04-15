#include "modelSpec.h"

void modelDefinition (ModelSpec& model) {
    model.setDT(1.0);
    model.setName("tutorial2");

    NeuronModels::Izhikevich::ParamValues izkParams(0.02, 0.2, -65.0, 8.0);
    InitVarSnippet::Uniform::ParamValues uDist(0.0, 30.0);

    NeuronModels::Izhikevich::VarValues ikzInit(-65.0, initVar<InitVarSnippet::Uniform>(uDist));

    model.addNeuronPopulation<NeuronModels::Izhikevich>("Exc", 8000, izkParams, ikzInit); // Excite - send to a synapse using the axon
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Inh", 2000, izkParams, ikzInit); // Inhibit - take in from a synapse through dendrite

    CurrentSourceModels::DC::ParamValues currentSourceParamValues(6.0);
    model.addCurrentSource<CurrentSourceModels::DC>("ExcStim", "Exc", currentSourceParamValues, {});
    model.addCurrentSource<CurrentSourceModels::DC>("InhStim", "Inh", currentSourceParamValues, {});

    WeightUpdateModels::StaticPulse::VarValues excSynInitVals(0.05);
    WeightUpdateModels::StaticPulse::VarValues inhSynInitVals(-5 * 0.05);

    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(0.1);

    // For all connections
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Exc_Exc", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY, "Exc", "Exc", {}, excSynInitVals, {}, {}, initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb)
    );
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Exc_Inh", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY, "Exc", "Inh", {}, excSynInitVals, {}, {}, initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb)
    );
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Inh_Inh", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY, "Inh", "Inh", {}, inhSynInitVals, {}, {}, initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb)
    );
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Inh_Exc", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY, "Inh", "Exc", {}, inhSynInitVals, {}, {}, initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb)
    );

}