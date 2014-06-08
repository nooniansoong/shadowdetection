/* 
 * File:   openCLtols.h
 * Author: marko
 *
 * Created on May 28, 2014, 10:10 PM
 */

#ifndef OPENCLTOLS_H
#define	OPENCLTOLS_H

#ifdef _OPENCL

#ifndef _MAC
#include <CL/cl.h>
#else
#include <OpenCL/opencl.h>
#endif
#include "typedefs.h"

#define MAX_DEVICES 100
#define MAX_SRC_SIZE 5242800
#define KERNEL_COUNT 3
#define MAX_PLATFORMS 100

namespace cv{
    class Mat;
}

namespace shadowdetection {
    namespace opencl {        

        class OpenclTools {
        private:            
            cl_device_id device;
            cl_int err;
            cl_command_queue command_queue;
            cl_program program;
            cl_kernel kernel[KERNEL_COUNT];
            cl_context context;
            size_t workGroupSize[KERNEL_COUNT];
            cl_mem input;
            cl_mem output1;
            cl_mem output2;
            cl_mem output3;
            unsigned char* ratios1;
            unsigned char* ratios2;
            
            void err_check(int err, std::string err_code) throw (SDException&);
            void createKernels();
            void createWorkGroupSizes();
            void createBuffers(unsigned char* image, u_int32_t height, u_int32_t width, unsigned char channels);            
            void initVars();
            void initWorkVars();
            void setKernelArgs1(u_int32_t height, u_int32_t width, unsigned char channels);
            void setKernelArgs2(u_int32_t height, u_int32_t width, unsigned char channels);
            void setKernelArgs3(u_int32_t height, u_int32_t width, unsigned char channels);
            
            void loadKernelFile(std::string kernelFileName);
            void loadKernelFileFromSource(std::string kernelFileName);
            bool loadKernelFileFromBinary(std::string kernelFileName);
            const char* saveKernelBinary(std::string kernelFileName);
        protected:
        public:
            OpenclTools();
            virtual ~OpenclTools();
            void init(int platformID, int deviceID, bool listOnly) throw (SDException&);
            cv::Mat* processRGBImage(unsigned char* image, u_int32_t width, u_int32_t height, unsigned char channels) throw (SDException&);
            cv::Mat* processRGBImageOld(unsigned char* image, u_int32_t width, u_int32_t height, unsigned char channels) throw (SDException&);
            void cleanUp();
            void cleanWorkPart();
        };

    }
}

#endif

#endif	/* OPENCLTOLS_H */

