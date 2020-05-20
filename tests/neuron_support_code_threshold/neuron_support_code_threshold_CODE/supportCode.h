#pragma once

// support code for neuron groups
namespace post_neuron {
    SUPPORT_CODE_FUNC bool checkThreshold(scalar x){ return (fmodf(x, 1.0f) < 1e-4f); }
}
namespace pre_neuron {
    SUPPORT_CODE_FUNC bool checkThreshold(scalar x){ return (fmodf(x, 1.0f) < 1e-4f); }
}

// support code for synapse groups

