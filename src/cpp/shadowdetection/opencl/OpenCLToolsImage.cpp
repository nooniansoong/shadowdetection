#ifdef _OPENCL
#include "OpenCLTools.h"
#include "shadowdetection/util/MemMenager.h"

namespace shadowdetection {
    namespace opencl {
        
        using namespace shadowdetection::util;
        
        uint32_t* OpenclTools::convertHSI1( uchar* image, u_int32_t width, u_int32_t height, 
                                            uchar channels) throw(SDException&){
            if (initialized == false){
                SDException exc(SHADOW_OPENCL_TOOLS_NOT_INITIALIZED, "OpenclTools::convertHSI1");
                throw exc;
            }
            createBuffers(image, height, width, channels);
            setKernelArgs1(height, width, channels, 0);
            size_t local_ws = workGroupSize[0];
            size_t global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(   command_queue[0], kernel[0], 1, 0, 
                                            &global_ws, &local_ws, 0, 0, 0);
            err_check(err, "clEnqueueNDRangeKernel OpenclTools::convertHSI1", -1);
            size_t size = width * height * channels; 
            uint32_t* retArr = MemMenager::allocate<uint32_t>(size);
            err = clEnqueueReadBuffer(  command_queue[0], hsi1Converted, CL_FALSE, 0, 
                                        size * sizeof(cl_uint), retArr, 0, 0, 0);
            err_check(err, "clEnqueueReadBuffer OpenclTools::convertHSI1", -1);
            clFlush(command_queue[0]);
            clFinish(command_queue[0]);
            cleanWorkPart();
            return retArr;
        }
        
        uint32_t* OpenclTools::convertHSI2( uchar* image, u_int32_t width, u_int32_t height, 
                                            uchar channels) throw(SDException&){
            if (initialized == false){
                SDException exc(SHADOW_OPENCL_TOOLS_NOT_INITIALIZED, "OpenclTools::convertHSI2");
                throw exc;
            }
            createBuffers(image, height, width, channels);
            setKernelArgs1(height, width, channels, 1);
            size_t local_ws = workGroupSize[1];
            size_t global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(   command_queue[0], kernel[1], 1, 0, 
                                            &global_ws, &local_ws, 0, 0, 0);
            err_check(err, "clEnqueueNDRangeKernel OpenclTools::convertHSI2", -1);
            size_t size = width * height * channels; 
            uint32_t* retArr = MemMenager::allocate<uint32_t>(size);
            err = clEnqueueReadBuffer(  command_queue[0], hsi1Converted, CL_FALSE, 0, 
                                        size * sizeof(cl_uint), retArr, 0, 0, 0);
            err_check(err, "clEnqueueReadBuffer OpenclTools::convertHSI1", -1);
            clFlush(command_queue[0]);
            clFinish(command_queue[0]);
            cleanWorkPart();
            return retArr;
        }
    }
}

#endif