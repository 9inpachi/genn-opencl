#include "definitionsInternal.h"


// Update neurons kernel
extern "C" const char* updateNeuronsKernelSource = R"(typedef float scalar;

__kernel void preNeuronResetKernel(__global unsigned int* glbSpkCntNeurons) {
	int groupId = get_group_id(0);
	int localId = get_local_id(0);
	unsigned int id = 32 * groupId + localId;
	if (id == 0) {
		glbSpkCntNeurons[0] = 0;
	}
}

__kernel void updateNeuronsKernel(const float t,
	const float DT,
	__global unsigned int* glbSpkCntNeurons,
	__global unsigned int* glbSpkNeurons,
	__global scalar* VNeurons,
	__global scalar* UNeurons,
	__global scalar* aNeurons,
	__global scalar* bNeurons,
	__global scalar* cNeurons,
	__global scalar* dNeurons)
{
	int groupId = get_group_id(0);
	int localId = get_local_id(0);
	const unsigned int id = 32 * groupId + localId;
	volatile __local unsigned int shSpk[32];
	volatile __local unsigned int shPosSpk;
	volatile __local unsigned int shSpkCount;
	if (localId == 0); {
		shSpkCount = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	// Neurons
	if (id < 32) {

		if (id < 7) {
			scalar lV = VNeurons[id];
			scalar lU = UNeurons[id];
			scalar la = aNeurons[id];
			scalar lb = bNeurons[id];
			scalar lc = cNeurons[id];
			scalar ld = dNeurons[id];

			float Isyn = 0;
			// current source CurrentSource
			{
				Isyn += (1.00000000000000000e+01f);

			}
			// test whether spike condition was fulfilled previously
			const bool oldSpike = (lV >= 29.99f);
			// calculate membrane potential
			if (lV >= 30.0f) {
				lV = lc;
				lU += ld;
			}
			lV += 0.5f * (0.04f * lV * lV + 5.0f * lV + 140.0f - lU + Isyn) * DT; //at two times for numerical stability
			lV += 0.5f * (0.04f * lV * lV + 5.0f * lV + 140.0f - lU + Isyn) * DT;
			lU += la * (lb * lV - lU) * DT;
			if (lV > 30.0f) {   //keep this to not confuse users with unrealistiv voltage values 
				lV = 30.0f;
			}

			// test for and register a true spike
			if ((lV >= 29.99f) && !(oldSpike)) {
				const unsigned int spkIdx = atomic_add(&shSpkCount, 1);
				shSpk[spkIdx] = id;
			}
			VNeurons[id] = lV;
			UNeurons[id] = lU;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localId == 0) {
			if (shSpkCount > 0) {
				shPosSpk = atomic_add(&glbSpkCntNeurons[0], shSpkCount);
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localId < shSpkCount) {
			const unsigned int n = shSpk[localId];
			glbSpkNeurons[shPosSpk + localId] = n;
		}
	}

})";

void updateNeurons(float t) {

	// preNeuronResetKernel

	// Creating a preNeuronResetQueue for running the preNeuronResetKernel
	commandQueue.enqueueNDRangeKernel(preNeuronResetKernel, cl::NullRange, cl::NDRange(32));
	commandQueue.finish();

	// updateNeuronsKernel
	
	// Setting kernel arguments
	updateNeuronsKernel.setArg(0, t);

	// Running the updateNeuronsKernel
	commandQueue.enqueueNDRangeKernel(updateNeuronsKernel, cl::NullRange, cl::NDRange(32));
	commandQueue.finish();
	
}