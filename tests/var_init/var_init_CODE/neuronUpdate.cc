#include "definitionsInternal.h"
#include "supportCode.h"

extern "C" const char* updateNeuronsProgramSrc = R"(// ------------------------------------------------------------------------
// C99 sized types
typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;

#define CLRNG_SINGLE_PRECISION
#include <clRNG/lfsr113.clh>
#include <clRNG/philox432.clh>
typedef float scalar;
#define DT 0.100000f
#define TIME_MIN 1.175494351e-38f
#define TIME_MAX 3.402823466e+38f

// ------------------------------------------------------------------------
// Non-uniform generators
inline float exponentialDistLfsr113(clrngLfsr113Stream *rng) {
    while (true) {
        const float u = clrngLfsr113RandomU01(rng);
        if (u != 0.000000f) {
            return -log(u);
        }
    }
}

inline float normalDistLfsr113(clrngLfsr113Stream *rng) {
    const float u1 = clrngLfsr113RandomU01(rng);
    const float u2 = clrngLfsr113RandomU01(rng);
    const float r = sqrt(-2.000000f * log(u1));
    const float theta = 2.000000f * M_PI_F * u2;
    return r * sin(theta);
}

inline float logNormalDistLfsr113(clrngLfsr113Stream *rng, float mean,float stddev)
 {
    return exp(mean + (stddev * normalDistLfsr113(rng)));
}

inline float gammaDistInternalLfsr113(clrngLfsr113Stream *rng, float c, float d)
 {
    float x, v, u;
    while (true) {
        do {
            x = normalDistLfsr113(rng);
            v = 1.000000f + c*x;
        }
        while (v <= 0.000000f);
        
        v = v*v*v;
        do {
            u = clrngLfsr113RandomU01(rng);
        }
        while (u == 1.000000f);
        
        if (u < 1.000000f - 0.033100f*x*x*x*x) break;
        if (log(u) < 0.500000f*x*x + d*(1.000000f - v + log(v))) break;
    }
    
    return d*v;
}

inline float gammaDistLfsr113(clrngLfsr113Stream *rng, float a)
 {
    if (a > 1)
     {
        const float u = clrngLfsr113RandomU01 (rng);
        const float d = (1.000000f + a) - 1.000000f / 3.000000f;
        const float c = (1.000000f / 3.000000f) / sqrt(d);
        return gammaDistInternalLfsr113(rng, c, d) * pow(u, 1.000000f / a);
    }
    else
     {
        const float d = a - 1.000000f / 3.000000f;
        const float c = (1.000000f / 3.000000f) / sqrt(d);
        return gammaDistInternalLfsr113(rng, c, d);
    }
}

inline float exponentialDistPhilox432(clrngPhilox432Stream *rng) {
    while (true) {
        const float u = clrngPhilox432RandomU01(rng);
        if (u != 0.000000f) {
            return -log(u);
        }
    }
}

inline float normalDistPhilox432(clrngPhilox432Stream *rng) {
    const float u1 = clrngPhilox432RandomU01(rng);
    const float u2 = clrngPhilox432RandomU01(rng);
    const float r = sqrt(-2.000000f * log(u1));
    const float theta = 2.000000f * M_PI_F * u2;
    return r * sin(theta);
}

inline float logNormalDistPhilox432(clrngPhilox432Stream *rng, float mean,float stddev)
 {
    return exp(mean + (stddev * normalDistPhilox432(rng)));
}

inline float gammaDistInternalPhilox432(clrngPhilox432Stream *rng, float c, float d)
 {
    float x, v, u;
    while (true) {
        do {
            x = normalDistPhilox432(rng);
            v = 1.000000f + c*x;
        }
        while (v <= 0.000000f);
        
        v = v*v*v;
        do {
            u = clrngPhilox432RandomU01(rng);
        }
        while (u == 1.000000f);
        
        if (u < 1.000000f - 0.033100f*x*x*x*x) break;
        if (log(u) < 0.500000f*x*x + d*(1.000000f - v + log(v))) break;
    }
    
    return d*v;
}

inline float gammaDistPhilox432(clrngPhilox432Stream *rng, float a)
 {
    if (a > 1)
     {
        const float u = clrngPhilox432RandomU01 (rng);
        const float d = (1.000000f + a) - 1.000000f / 3.000000f;
        const float c = (1.000000f / 3.000000f) / sqrt(d);
        return gammaDistInternalPhilox432(rng, c, d) * pow(u, 1.000000f / a);
    }
    else
     {
        const float d = a - 1.000000f / 3.000000f;
        const float c = (1.000000f / 3.000000f) / sqrt(d);
        return gammaDistInternalPhilox432(rng, c, d);
    }
}

__kernel void preNeuronResetKernel(__global unsigned int* d_glbSpkCntPop, __global unsigned int* d_glbSpkCntSpikeSource) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    if(id == 0) {
        d_glbSpkCntPop[0] = 0;
    }
    else if(id == 1) {
        d_glbSpkCntSpikeSource[0] = 0;
    }
}

__kernel void updateNeuronsKernel(__global scalar* d_constantCurrSource, __global scalar* d_constantPop, __global scalar* d_exponentialCurrSource, __global scalar* d_exponentialPop, __global scalar* d_gammaCurrSource, __global scalar* d_gammaPop, __global unsigned int* d_glbSpkCntSpikeSource, __global unsigned int* d_glbSpkSpikeSource, __global scalar* d_inSynDense, __global scalar* d_inSynSparse, __global scalar* d_normalCurrSource, __global scalar* d_normalPop, __global scalar* d_pconstantDense, __global scalar* d_pconstantSparse, __global scalar* d_pexponentialDense, __global scalar* d_pexponentialSparse, __global scalar* d_pgammaDense, __global scalar* d_pgammaSparse, __global scalar* d_pnormalDense, __global scalar* d_pnormalSparse, __global scalar* d_puniformDense, __global scalar* d_puniformSparse, __global scalar* d_uniformCurrSource, __global scalar* d_uniformPop, float t) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    volatile __local unsigned int shSpk[32];
    volatile __local unsigned int shPosSpk;
    volatile __local unsigned int shSpkCount;
    if (localId == 0) {
        shSpkCount = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    // Pop
    if(id < 10016) {
        
        if(id < 10000) {
            scalar lconstant = d_constantPop[id];
            scalar luniform = d_uniformPop[id];
            scalar lnormal = d_normalPop[id];
            scalar lexponential = d_exponentialPop[id];
            scalar lgamma = d_gammaPop[id];
            
            // pull inSyn values in a coalesced access
            float linSynSparse = d_inSynSparse[id];
            scalar lpspconstantSparse = d_pconstantSparse[id];
            scalar lpspuniformSparse = d_puniformSparse[id];
            scalar lpspnormalSparse = d_pnormalSparse[id];
            scalar lpspexponentialSparse = d_pexponentialSparse[id];
            scalar lpspgammaSparse = d_pgammaSparse[id];
            
            // pull inSyn values in a coalesced access
            float linSynDense = d_inSynDense[id];
            scalar lpspconstantDense = d_pconstantDense[id];
            scalar lpspuniformDense = d_puniformDense[id];
            scalar lpspnormalDense = d_pnormalDense[id];
            scalar lpspexponentialDense = d_pexponentialDense[id];
            scalar lpspgammaDense = d_pgammaDense[id];
            
            // current source CurrSource
             {
                scalar lcsconstant = d_constantCurrSource[id];
                scalar lcsuniform = d_uniformCurrSource[id];
                scalar lcsnormal = d_normalCurrSource[id];
                scalar lcsexponential = d_exponentialCurrSource[id];
                scalar lcsgamma = d_gammaCurrSource[id];
                
                d_constantCurrSource[id] = lcsconstant;
                d_uniformCurrSource[id] = lcsuniform;
                d_normalCurrSource[id] = lcsnormal;
                d_exponentialCurrSource[id] = lcsexponential;
                d_gammaCurrSource[id] = lcsgamma;
            }
            // calculate membrane potential
            
            d_constantPop[id] = lconstant;
            d_uniformPop[id] = luniform;
            d_normalPop[id] = lnormal;
            d_exponentialPop[id] = lexponential;
            d_gammaPop[id] = lgamma;
            // the post-synaptic dynamics
            
            d_inSynSparse[id] = linSynSparse;
            d_pconstantSparse[id] = lpspconstantSparse;
            d_puniformSparse[id] = lpspuniformSparse;
            d_pnormalSparse[id] = lpspnormalSparse;
            d_pexponentialSparse[id] = lpspexponentialSparse;
            d_pgammaSparse[id] = lpspgammaSparse;
            // the post-synaptic dynamics
            
            d_inSynDense[id] = linSynDense;
            d_pconstantDense[id] = lpspconstantDense;
            d_puniformDense[id] = lpspuniformDense;
            d_pnormalDense[id] = lpspnormalDense;
            d_pexponentialDense[id] = lpspexponentialDense;
            d_pgammaDense[id] = lpspgammaDense;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // SpikeSource
    if(id >= 10016 && id < 10048) {
        const unsigned int lid = id - 10016;
        
        if(lid < 1) {
            
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            // test for and register a true spike
            if (0) {
                const unsigned int spkIdx = atomic_add(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomic_add(&d_glbSpkCntSpikeSource[0], shSpkCount);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < shSpkCount) {
            const unsigned int n = shSpk[localId];
            d_glbSpkSpikeSource[shPosSpk + localId] = n;
        }
    }
    
}
)";

// Initialize the neuronUpdate kernels
void updateNeuronsProgramKernels() {
    preNeuronResetKernel = cl::Kernel(updateNeuronsProgram, "preNeuronResetKernel");
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(0, d_glbSpkCntPop));
    CHECK_OPENCL_ERRORS(preNeuronResetKernel.setArg(1, d_glbSpkCntSpikeSource));
    
    updateNeuronsKernel = cl::Kernel(updateNeuronsProgram, "updateNeuronsKernel");
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(0, d_constantCurrSource));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(1, d_constantPop));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(2, d_exponentialCurrSource));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(3, d_exponentialPop));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(4, d_gammaCurrSource));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(5, d_gammaPop));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(6, d_glbSpkCntSpikeSource));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(7, d_glbSpkSpikeSource));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(8, d_inSynDense));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(9, d_inSynSparse));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(10, d_normalCurrSource));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(11, d_normalPop));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(12, d_pconstantDense));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(13, d_pconstantSparse));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(14, d_pexponentialDense));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(15, d_pexponentialSparse));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(16, d_pgammaDense));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(17, d_pgammaSparse));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(18, d_pnormalDense));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(19, d_pnormalSparse));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(20, d_puniformDense));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(21, d_puniformSparse));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(22, d_uniformCurrSource));
    CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(23, d_uniformPop));
}

void updateNeurons(float t) {
     {
        const cl::NDRange globalWorkSize(32, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
        
    }
     {
        CHECK_OPENCL_ERRORS(updateNeuronsKernel.setArg(24, t));
        
        const cl::NDRange globalWorkSize(10048, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
