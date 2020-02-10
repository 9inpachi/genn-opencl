#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>
#include <cassert>
#include <fstream>

#define EXPORT_VAR extern
#define EXPORT_FUNC

typedef float scalar;

#define DT 0.100000f
#define NSIZE 7

extern "C" {
	EXPORT_VAR unsigned int* dd_glbSpkCntNeurons;
	EXPORT_VAR unsigned int* dd_glbSpkNeurons;
	EXPORT_VAR scalar* dd_VNeurons;
	EXPORT_VAR scalar* dd_UNeurons;
	EXPORT_VAR scalar* dd_aNeurons;
	EXPORT_VAR scalar* dd_bNeurons;
	EXPORT_VAR scalar* dd_cNeurons;
	EXPORT_VAR scalar* dd_dNeurons;
	EXPORT_VAR unsigned long long iT;
	EXPORT_VAR float t;

	EXPORT_FUNC void updateNeurons(float t);
	EXPORT_FUNC void allocateMem();
	EXPORT_FUNC void initialize();
	EXPORT_FUNC void stepTime();
	EXPORT_FUNC scalar* getCurrentVNeurons();

}










/*

INDEPENDENT EXTRA FUNCTIONS

*/



namespace OpenCLModule {
	class OpenCLProgram {
	private:
		void createProgram(const std::string& filename, cl::Program& program, cl::Context& context, std::vector<cl::Device>& devices, cl::Device& device, const int deviceIndex) {

			// Getting all platforms to gather devices from
			std::vector<cl::Platform> platforms;
			cl::Platform::get(&platforms); // Gets all the platforms

			assert(platforms.size() > 0);

			// Getting all devices and putting them into a single vector
			for (int i = 0; i < platforms.size(); i++) {
				std::vector<cl::Device> platformDevices;
				platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
				devices.insert(devices.end(), platformDevices.begin(), platformDevices.end());
			}

			assert(devices.size() > 0);

			// Check if the device exists at the given index
			if (deviceIndex >= devices.size()) {
				assert(deviceIndex >= devices.size());
				device = devices.front();
			}
			else {
				device = devices[deviceIndex]; // We will perform our operations using this device
			}

			// Reading the kernel file for execution
			std::ifstream kernelFile(filename);
			std::string srcString(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));


			cl::Program::Sources programSources(1, std::make_pair(srcString.c_str(), srcString.length() + 1));

			context = cl::Context(device);
			program = cl::Program(context, programSources);

			program.build("-cl-std=CL1.2");

		}

	public:
		cl::Program program;
		cl::Context context;
		cl::Device device;
		int deviceIndex;
		std::vector<cl::Device> allDevices;

		OpenCLProgram(){}
		OpenCLProgram(const std::string& kernelFile, const int deviceIndex) {
			this->createProgram(kernelFile, program, context, allDevices, device, deviceIndex);
		}

		void runKernel(const char* kernelName, std::vector<std::vector<int>>& valueVec) {
			std::vector<cl::Buffer> bufVec(valueVec.size());
			for (int i = 0; i < bufVec.size(); i++) {
				bufVec[i] = cl::Buffer(context, valueVec[i].begin(), valueVec[i].end(), true);
			}

			cl::Kernel kernel(program, kernelName);
			for (int i = 0; i < bufVec.size(); i++) {
				kernel.setArg(i, bufVec[i]);
			}

			cl::CommandQueue queue(context, device);
			queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(valueVec[0].size()));
			queue.enqueueReadBuffer(bufVec[0], CL_TRUE, 0, sizeof(int) * valueVec[0].size(), valueVec[0].data());

			queue.finish();
		}

		std::string getCLError(cl_int errorCode) {
			switch (errorCode) {
			case CL_SUCCESS:                            return "Success!";
			case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
			case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
			case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
			case CL_OUT_OF_RESOURCES:                   return "Out of resources";
			case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
			case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
			case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
			case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
			case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
			case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
			case CL_MAP_FAILURE:                        return "Map failure";
			case CL_INVALID_VALUE:                      return "Invalid value";
			case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
			case CL_INVALID_PLATFORM:                   return "Invalid platform";
			case CL_INVALID_DEVICE:                     return "Invalid device";
			case CL_INVALID_CONTEXT:                    return "Invalid context";
			case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
			case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
			case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
			case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
			case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
			case CL_INVALID_SAMPLER:                    return "Invalid sampler";
			case CL_INVALID_BINARY:                     return "Invalid binary";
			case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
			case CL_INVALID_PROGRAM:                    return "Invalid program";
			case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
			case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
			case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
			case CL_INVALID_KERNEL:                     return "Invalid kernel";
			case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
			case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
			case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
			case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
			case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
			case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
			case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
			case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
			case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
			case CL_INVALID_EVENT:                      return "Invalid event";
			case CL_INVALID_OPERATION:                  return "Invalid operation";
			case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
			case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
			case CL_INVALID_MIP_LEVEL:                  return "Invalid mip - map level";
			default: return "Unknown";
			}
		}

	};
}
