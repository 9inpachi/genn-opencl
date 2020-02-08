#include "definitions.h"

extern "C" {
	std::vector<unsigned int> dd_glbSpkCntNeurons(1); // __device__
	std::vector<unsigned int> dd_glbSpkNeurons(7); // __device__
	std::vector<scalar> dd_VNeurons(7); // __device__
	std::vector<scalar> dd_UNeurons(7); // __device__
	std::vector<scalar> dd_aNeurons(7); // __device__
	std::vector<scalar> dd_bNeurons(7); // __device__
	std::vector<scalar> dd_cNeurons(7); // __device__
	std::vector<scalar> dd_dNeurons(7); // __device__
	std::vector<scalar> aNeurons(7);
	std::vector<scalar> bNeurons(7);
	std::vector<scalar> cNeurons(7);
	std::vector<scalar> dNeurons(7);
}