#include "definitionsInternal.h"


extern "C" const char* initProgramSrc = R"(// ------------------------------------------------------------------------
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

__kernel void initializeKernel(__global scalar* d_constantCurrSource, __global scalar* d_constantDense, __global scalar* d_constantPop, __global scalar* d_exponentialCurrSource, __global scalar* d_exponentialDense, __global scalar* d_exponentialPop, __global scalar* d_gammaCurrSource, __global scalar* d_gammaDense, __global scalar* d_gammaPop, __global unsigned int* d_glbSpkCntPop, __global unsigned int* d_glbSpkCntSpikeSource, __global unsigned int* d_glbSpkPop, __global unsigned int* d_glbSpkSpikeSource, __global scalar* d_inSynDense, __global scalar* d_inSynSparse, __global scalar* d_normalCurrSource, __global scalar* d_normalDense, __global scalar* d_normalPop, __global scalar* d_pconstantDense, __global scalar* d_pconstantSparse, __global scalar* d_pexponentialDense, __global scalar* d_pexponentialSparse, __global scalar* d_pgammaDense, __global scalar* d_pgammaSparse, __global scalar* d_pnormalDense, __global scalar* d_pnormalSparse, __global scalar* d_puniformDense, __global scalar* d_puniformSparse, __global clrngPhilox432HostStream* d_rng, __global scalar* d_uniformCurrSource, __global scalar* d_uniformDense, __global scalar* d_uniformPop, unsigned int deviceRNGSeed) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    // ------------------------------------------------------------------------
    // Local neuron groups
    // Pop
    if(id < 10016) {
        // only do this for existing neurons
        if(id < 10000) {
            clrngPhilox432Stream localStream;
            clrngPhilox432CopyOverStreamsFromGlobal(1, &localStream, &d_rng[0]);
            const clrngPhilox432Counter steps = {{0, id}, {0, 0}};
            localStream.current.ctr = clrngPhilox432Add(localStream.current.ctr, steps);
            localStream.current.deckIndex = 0;
            clrngPhilox432GenerateDeck(&localStream.current);
            if(id == 0) {
                d_glbSpkCntPop[0] = 0;
            }
            d_glbSpkPop[id] = 0;
             {
                d_constantPop[id] = (1.30000000000000000e+01f);
            }
             {
                const scalar scale = (1.00000000000000000e+00f) - (0.00000000000000000e+00f);
                d_uniformPop[id] = (0.00000000000000000e+00f) + (clrngPhilox432RandomU01(&localStream) * scale);
            }
             {
                d_normalPop[id] = (0.00000000000000000e+00f) + (normalDistPhilox432(&localStream) * (1.00000000000000000e+00f));
            }
             {
                d_exponentialPop[id] = (1.00000000000000000e+00f) * exponentialDistPhilox432(&localStream);
            }
             {
                d_gammaPop[id] = (1.00000000000000000e+00f) * gammaDistPhilox432(&localStream, (4.00000000000000000e+00f));
            }
            d_inSynSparse[id] = 0.000000f;
             {
                d_pconstantSparse[id] = (1.30000000000000000e+01f);
            }
             {
                const scalar scale = (1.00000000000000000e+00f) - (0.00000000000000000e+00f);
                d_puniformSparse[id] = (0.00000000000000000e+00f) + (clrngPhilox432RandomU01(&localStream) * scale);
            }
             {
                d_pnormalSparse[id] = (0.00000000000000000e+00f) + (normalDistPhilox432(&localStream) * (1.00000000000000000e+00f));
            }
             {
                d_pexponentialSparse[id] = (1.00000000000000000e+00f) * exponentialDistPhilox432(&localStream);
            }
             {
                d_pgammaSparse[id] = (1.00000000000000000e+00f) * gammaDistPhilox432(&localStream, (4.00000000000000000e+00f));
            }
            d_inSynDense[id] = 0.000000f;
             {
                d_pconstantDense[id] = (1.30000000000000000e+01f);
            }
             {
                const scalar scale = (1.00000000000000000e+00f) - (0.00000000000000000e+00f);
                d_puniformDense[id] = (0.00000000000000000e+00f) + (clrngPhilox432RandomU01(&localStream) * scale);
            }
             {
                d_pnormalDense[id] = (0.00000000000000000e+00f) + (normalDistPhilox432(&localStream) * (1.00000000000000000e+00f));
            }
             {
                d_pexponentialDense[id] = (1.00000000000000000e+00f) * exponentialDistPhilox432(&localStream);
            }
             {
                d_pgammaDense[id] = (1.00000000000000000e+00f) * gammaDistPhilox432(&localStream, (4.00000000000000000e+00f));
            }
            // current source variables
             {
                d_constantCurrSource[id] = (1.30000000000000000e+01f);
            }
             {
                const scalar scale = (1.00000000000000000e+00f) - (0.00000000000000000e+00f);
                d_uniformCurrSource[id] = (0.00000000000000000e+00f) + (clrngPhilox432RandomU01(&localStream) * scale);
            }
             {
                d_normalCurrSource[id] = (0.00000000000000000e+00f) + (normalDistPhilox432(&localStream) * (1.00000000000000000e+00f));
            }
             {
                d_exponentialCurrSource[id] = (1.00000000000000000e+00f) * exponentialDistPhilox432(&localStream);
            }
             {
                d_gammaCurrSource[id] = (1.00000000000000000e+00f) * gammaDistPhilox432(&localStream, (4.00000000000000000e+00f));
            }
        }
    }
    
    // SpikeSource
    if(id >= 10016 && id < 10048) {
        const unsigned int lid = id - 10016;
        // only do this for existing neurons
        if(lid < 1) {
            if(lid == 0) {
                d_glbSpkCntSpikeSource[0] = 0;
            }
            d_glbSpkSpikeSource[lid] = 0;
            // current source variables
        }
    }
    
    
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    // Dense
    if(id >= 10048 && id < 20064) {
        const unsigned int lid = id - 10048;
        // only do this for existing postsynaptic neurons
        if(lid < 10000) {
            clrngPhilox432Stream localStream;
            clrngPhilox432CopyOverStreamsFromGlobal(1, &localStream, &d_rng[0]);
            const clrngPhilox432Counter steps = {{0, id}, {0, 0}};
            localStream.current.ctr = clrngPhilox432Add(localStream.current.ctr, steps);
            localStream.current.deckIndex = 0;
            clrngPhilox432GenerateDeck(&localStream.current);
            for(unsigned int i = 0; i < 1; i++) {
                 {
                    d_constantDense[(i * 10000) + lid] = (1.30000000000000000e+01f);
                }
                 {
                    const scalar scale = (1.00000000000000000e+00f) - (0.00000000000000000e+00f);
                    d_uniformDense[(i * 10000) + lid] = (0.00000000000000000e+00f) + (clrngPhilox432RandomU01(&localStream) * scale);
                }
                 {
                    d_normalDense[(i * 10000) + lid] = (0.00000000000000000e+00f) + (normalDistPhilox432(&localStream) * (1.00000000000000000e+00f));
                }
                 {
                    d_exponentialDense[(i * 10000) + lid] = (1.00000000000000000e+00f) * exponentialDistPhilox432(&localStream);
                }
                 {
                    d_gammaDense[(i * 10000) + lid] = (1.00000000000000000e+00f) * gammaDistPhilox432(&localStream, (4.00000000000000000e+00f));
                }
            }
        }
    }
    
    
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
}

__kernel void initializeSparseKernel(__global scalar* d_constantSparse, __global scalar* d_exponentialSparse, __global scalar* d_gammaSparse, __global unsigned int* d_indSparse, __global scalar* d_normalSparse, __global scalar* d_pconstantDense, __global scalar* d_pconstantSparse, __global scalar* d_pexponentialDense, __global scalar* d_pexponentialSparse, __global scalar* d_pgammaDense, __global scalar* d_pgammaSparse, __global scalar* d_pnormalDense, __global scalar* d_pnormalSparse, __global scalar* d_puniformDense, __global scalar* d_puniformSparse, __global clrngPhilox432HostStream* d_rng, __global unsigned int* d_rowLengthSparse, __global scalar* d_uniformSparse) {
    const size_t localId = get_local_id(0);
    const unsigned int id = get_global_id(0);
    __local unsigned int shRowLength[32];
    // Sparse
    if(id < 10016) {
        clrngPhilox432Stream localStream;
        clrngPhilox432CopyOverStreamsFromGlobal(1, &localStream, &d_rng[0]);
        const clrngPhilox432Counter steps = {{0, id}, {0, 0}};
        localStream.current.ctr = clrngPhilox432Add(localStream.current.ctr, steps);
        localStream.current.deckIndex = 0;
        clrngPhilox432GenerateDeck(&localStream.current);
        unsigned int idx = id;
        for(unsigned int r = 0; r < 1; r++) {
            const unsigned numRowsInBlock = (r == 0) ? 1 : 32;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (localId < numRowsInBlock) {
                shRowLength[localId] = d_rowLengthSparse[(r * 32) + localId];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for(unsigned int i = 0; i < numRowsInBlock; i++) {
                if(id < shRowLength[i]) {
                     {
                        d_constantSparse[(((r * 32) + i) * 10000) + id] = (1.30000000000000000e+01f);
                    }
                     {
                        const scalar scale = (1.00000000000000000e+00f) - (0.00000000000000000e+00f);
                        d_uniformSparse[(((r * 32) + i) * 10000) + id] = (0.00000000000000000e+00f) + (clrngPhilox432RandomU01(&localStream) * scale);
                    }
                     {
                        d_normalSparse[(((r * 32) + i) * 10000) + id] = (0.00000000000000000e+00f) + (normalDistPhilox432(&localStream) * (1.00000000000000000e+00f));
                    }
                     {
                        d_exponentialSparse[(((r * 32) + i) * 10000) + id] = (1.00000000000000000e+00f) * exponentialDistPhilox432(&localStream);
                    }
                     {
                        d_gammaSparse[(((r * 32) + i) * 10000) + id] = (1.00000000000000000e+00f) * gammaDistPhilox432(&localStream, (4.00000000000000000e+00f));
                    }
                }
                idx += 10000;
            }
        }
    }
    
    
}

)";

// Initialize the initialization kernel(s)
void initProgramKernels() {
    initializeKernel = cl::Kernel(initProgram, "initializeKernel");
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(0, d_constantCurrSource));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(1, d_constantDense));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(2, d_constantPop));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(3, d_exponentialCurrSource));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(4, d_exponentialDense));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(5, d_exponentialPop));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(6, d_gammaCurrSource));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(7, d_gammaDense));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(8, d_gammaPop));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(9, d_glbSpkCntPop));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(10, d_glbSpkCntSpikeSource));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(11, d_glbSpkPop));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(12, d_glbSpkSpikeSource));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(13, d_inSynDense));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(14, d_inSynSparse));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(15, d_normalCurrSource));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(16, d_normalDense));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(17, d_normalPop));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(18, d_pconstantDense));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(19, d_pconstantSparse));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(20, d_pexponentialDense));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(21, d_pexponentialSparse));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(22, d_pgammaDense));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(23, d_pgammaSparse));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(24, d_pnormalDense));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(25, d_pnormalSparse));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(26, d_puniformDense));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(27, d_puniformSparse));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(28, d_rng));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(29, d_uniformCurrSource));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(30, d_uniformDense));
    CHECK_OPENCL_ERRORS(initializeKernel.setArg(31, d_uniformPop));
    
    initializeSparseKernel = cl::Kernel(initProgram, "initializeSparseKernel");
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(0, d_constantSparse));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(1, d_exponentialSparse));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(2, d_gammaSparse));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(3, d_indSparse));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(4, d_normalSparse));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(5, d_pconstantDense));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(6, d_pconstantSparse));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(7, d_pexponentialDense));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(8, d_pexponentialSparse));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(9, d_pgammaDense));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(10, d_pgammaSparse));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(11, d_pnormalDense));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(12, d_pnormalSparse));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(13, d_puniformDense));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(14, d_puniformSparse));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(15, d_rng));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(16, d_rowLengthSparse));
    CHECK_OPENCL_ERRORS(initializeSparseKernel.setArg(17, d_uniformSparse));
}

void initialize() {
     {
        unsigned int deviceRNGSeed = 0;
        
        CHECK_OPENCL_ERRORS(initializeKernel.setArg(32, deviceRNGSeed));
        
        const cl::NDRange globalWorkSize(20064, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(initializeKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}

// Initialize all OpenCL elements
void initializeSparse() {
    copyStateToDevice(true);
    copyConnectivityToDevice(true);
     {
        const cl::NDRange globalWorkSize(10016, 1);
        const cl::NDRange localWorkSize(32, 1);
        CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(initializeSparseKernel, cl::NullRange, globalWorkSize, localWorkSize));
        CHECK_OPENCL_ERRORS(commandQueue.finish());
    }
}
