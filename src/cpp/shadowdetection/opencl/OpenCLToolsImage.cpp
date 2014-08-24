#ifdef _OPENCL
#include "OpenCLTools.h"
#include "core/util/MemMenager.h"

namespace shadowdetection {
    namespace opencl {
        
        using namespace core::util;
        
        /**!!!!!!!NOT TESTED*/
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
            err = clEnqueueNDRangeKernel(   command_queue, kernel[0], 1, 0, 
                                            &global_ws, &local_ws, 0, 0, 0);
            err_check(err, "OpenclTools::convertHSI1 clEnqueueNDRangeKernel");
            size_t size = width * height * channels; 
            uint32_t* retArr = MemMenager::allocate<uint32_t>(size);
            err = clEnqueueReadBuffer(  command_queue, hsi1Converted, CL_FALSE, 0, 
                                        size * sizeof(cl_uint), retArr, 0, 0, 0);
            err_check(err, "OpenclTools::convertHSI1 clEnqueueReadBuffer");
            clFlush(command_queue);
            clFinish(command_queue);
            cleanWorkPart();
            return retArr;
        }
        
        /**!!!!!!!NOT TESTED*/
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
            err = clEnqueueNDRangeKernel(   command_queue, kernel[1], 1, 0, 
                                            &global_ws, &local_ws, 0, 0, 0);
            err_check(err, "OpenclTools::convertHSI2 clEnqueueNDRangeKernel");
            size_t size = width * height * channels; 
            uint32_t* retArr = MemMenager::allocate<uint32_t>(size);
            err = clEnqueueReadBuffer(  command_queue, hsi1Converted, CL_FALSE, 0, 
                                        size * sizeof(cl_uint), retArr, 0, 0, 0);
            err_check(err, "OpenclTools::convertHSI2 clEnqueueReadBuffer");
            clFlush(command_queue);
            clFinish(command_queue);
            cleanWorkPart();
            return retArr;
        }
    }
}

#endif