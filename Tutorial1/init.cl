// Will have to read the arguments again and update for the program
__kernel void initializeKernel(unsigned long* deviceRNGSeed,
__global unsigned int* dd_glbSpkCntNeurons,
__global unsigned int* dd_glbSpkNeurons,
__global float* dd_VNeurons,
__global float* dd_UNeurons
) {
	int groupId = get_group_id(0);
	int localId = get_local_id(0);
    const unsigned int id = 32 * groupId + localId;
    
    if(id < 32) {
        // only do this for existing neurons
        if(id < 7) {
            if(id == 0) {
                dd_glbSpkCntNeurons[0] = 0;
            }
            dd_glbSpkNeurons[id] = 0;
             {
                dd_VNeurons[id] = (-6.50000000000000000e+01f);
            }
             {
                dd_UNeurons[id] = (-2.00000000000000000e+01f);
            }
            // current source variables
        }
    }
}
