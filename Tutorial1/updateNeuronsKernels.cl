__kernel void preNeuronResetKernel(__global unsigned int* dd_glbSpkCntNeurons) {
	int groupId = get_group_id(0);
	int localId = get_local_id(0);
	unsigned int id = 32 * groupId + localId;
	if (id == 0) {
		dd_glbSpkCntNeurons[0] = 0;
	}
}