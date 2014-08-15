#ifdef _OPENCL
#include "OpenCLTools.h"

#include "core/opencv/OpenCV2Tools.h"
#include "core/opencv/OpenCVTools.h"
#include "thirdparty/lib_svm/svm.h"
#include "core/util/Matrix.h"

#define KERNEL_FILE_1 "image_hci_convert_kernel"
#define KERNEL_FILE_2 "lib_svm"
#define KERNEL_FILE_3 "lib_svm_predict"
#define KERNEL_PATH "src/cpp/shadowdetection/opencl/kernels/"


namespace shadowdetection {
    namespace opencl {        
        using namespace core::opencv2;
        using namespace core::opencv;
        using namespace cv;
        using namespace core::util;
        using namespace core::opencl;
        
        void OpenclTools::initVars(){          
            OpenClBase::initVars();
            
            //libsvm part
            clY                 = 0;
            clX                 = 0;            
            clData              = 0;
            newTask             = true;
            newSelectWorkingSet = true;
            clXSquared          = 0;
            xMatrix             = 0;
            
            durrData        = 0l;
            durrBuff        = 0l;
            durrExec        = 0l;
            durrReadBuff    = 0l;
            durrSetSrgs     = 0l;            
            
            //training part
            clQD                = 0;
            clAlphaStatus       = 0;
            clYSelectWorkingSet = 0;
            clG                 = 0;
            
            //predict part
            modelSVs        = 0;
            clModelSVs      = 0;
            clModelRHO      = 0;
            clModelSVCoefs  = 0;
            clModelLabel    = 0;
            svCoefs         = 0;
            clModelNsv      = 0;
            modelRHOs       = 0;
            
            initWorkVars();
        }
        
        void OpenclTools::initWorkVars(){
            inputImage = 0;
            hsi1Converted = 0;
            hsi2Converted = 0;
            tsaiOutput = 0;
            ratios1 = 0;
            ratios2 = 0;
            
            //train part            
            clGradDiff = 0;
            clObjDiff = 0;                        
            clQI = 0;            
            
            //predict part
            clPixelParameters = 0;
            clPredictResults = 0;
        }
        
        OpenclTools::OpenclTools() : Singleton<OpenclTools>(){            
            initialized = false;
            modelChanged = true;
            initVars();            
        }

        void OpenclTools::cleanWorkPart() {
            if (inputImage)
                clReleaseMemObject(inputImage);
            if (hsi1Converted)
                clReleaseMemObject(hsi1Converted);
            if (hsi2Converted)
                clReleaseMemObject(hsi2Converted);
            if (tsaiOutput)
                clReleaseMemObject(tsaiOutput);            

            if (ratios1)
                MemMenager::delocate(ratios1);
            if (ratios2)
                MemMenager::delocate(ratios2);
            
            //train part
            if (clGradDiff){
                clReleaseMemObject(clGradDiff);
                err_check(err, "OpenclTools::cleanWorkPart clGradDiff");
            }
            if (clObjDiff){
                clReleaseMemObject(clObjDiff);
                err_check(err, "OpenclTools::cleanWorkPart clObjDiff");
            }
            if (clQI){
                clReleaseMemObject(clQI);
                err_check(err, "OpenclTools::cleanWorkPart clQI");
            }
            
            //predict part
            if (clPixelParameters){
                clReleaseMemObject(clPixelParameters);
                err_check(err, "OpenclTools::cleanWorkPart clPredictResults");
            }
            if (clPredictResults){
                err = clReleaseMemObject(clPredictResults);
                err_check(err, "OpenclTools::cleanWorkPart clPredictResults");
            }    
            initWorkVars();           
        }

        void OpenclTools::cleanUp(){
            OpenClBase::cleanUp();
            
            //train part
            if (clY){
                err = clReleaseMemObject(clY);
                err_check(err, "OpenclTools::cleanUp clReleaseMemObject2");
            }
                
            if (clX){
                err = clReleaseMemObject(clX);
                err_check(err, "OpenclTools::cleanUp clReleaseMemObject3");
            }
                        
            if (clData){
                err = clReleaseMemObject(clData);
                err_check(err, "OpenclTools::cleanUp clReleaseMemObject1");
            }
            
            if (clXSquared){
                err = clReleaseMemObject(clXSquared);
                err_check(err, "OpenclTools::cleanUp clReleaseMemObjectXSquared");
            }
            
            if (xMatrix){
                delete xMatrix;
            }                        
            
            if (clQD){
                clReleaseMemObject(clQD);
                err_check(err, "OpenclTools::cleanUp clQD");
            }
            
            if (clAlphaStatus){
                clReleaseMemObject(clAlphaStatus);
                err_check(err, "OpenclTools::cleanUp clAlphaStatus");
            }
            
            if (clYSelectWorkingSet){
                clReleaseMemObject(clYSelectWorkingSet);
                err_check(err, "OpenclTools::cleanUp clYSelectWorkingSet");
            }
            
            if (clG){
                clReleaseMemObject(clG);
                err_check(err, "OpenclTools::cleanWorkPart clG");
            } 
            
            durrData = 0l;
            durrBuff = 0l;
            durrExec = 0l;
            durrReadBuff = 0l;
            durrSetSrgs = 0l;
            
            //predict part
            if (modelSVs)
                delete modelSVs;
            if (clModelSVs){
                err = clReleaseMemObject(clModelSVs);
                err_check(err, "OpenclTools::cleanUp clModelSVs");
            }
            if (clModelRHO){
                err = clReleaseMemObject(clModelRHO);
                err_check(err, "OpenclTools::cleanUp clModelRHO");
            }
            if (clModelSVCoefs){
                err = clReleaseMemObject(clModelSVCoefs);
                err_check(err, "OpenclTools::cleanUp clModelSVCoefs");
            }            
            if (clModelLabel){
                err = clReleaseMemObject(clModelLabel);
                err_check(err, "OpenclTools::cleanUp clModelLabel");
            }
            if (svCoefs)
                delete svCoefs;            
            if (clModelNsv){
                err = clReleaseMemObject(clModelNsv);
                err_check(err, "OpenclTools::cleanUp clModelNsv");
            }
            if (modelRHOs)
                MemMenager::delocate(modelRHOs);
            modelChanged = true;
                        
            cleanWorkPart();            
            initVars();            
        }

        OpenclTools::~OpenclTools(){            
        }                                                

        void OpenclTools::createBuffers(uchar* image, u_int32_t height, u_int32_t width, uchar channels) {
            size_t size = width * height * channels;
            cl_device_type type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
            if (type == CL_DEVICE_TYPE_GPU)
            {
                inputImage = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, image, &err);
                err_check(err, "OpenclTools::createBuffers inputImage");
                hsi1Converted = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof (u_int32_t), 0, &err);
                err_check(err, "OpenclTools::createBuffers hsi1Converted");
                hsi2Converted = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof (u_int32_t), 0, &err);
                err_check(err, "OpenclTools::createBuffers hsi2Converted");
                tsaiOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height, 0, &err);
                err_check(err, "OpenclTools::createBuffers tsaiOutput");
            }
            else if (type == CL_DEVICE_TYPE_CPU){
                int flag = CL_MEM_USE_HOST_PTR;
                inputImage = clCreateBuffer(context, CL_MEM_READ_ONLY | flag, size, image, &err);
                err_check(err, "OpenclTools::createBuffers inputImage");
                hsi1Converted = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof (u_int32_t), 0, &err);
                err_check(err, "OpenclTools::createBuffers hsi1Converted");
                hsi2Converted = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof (u_int32_t), 0, &err);
                err_check(err, "OpenclTools::createBuffers hsi2Converted");
                tsaiOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, width * height, 0, &err);
                err_check(err, "OpenclTools::createBuffers tsaiOutput");
            }
            else{
                SDException exc(SHADOW_NOT_SUPPORTED_DEVICE, 
                                "OpenclTools::createBuffers Init buffers, currently not supported device");
                throw exc;
            }
        }
        
        void OpenclTools::setKernelArgs1(   u_int32_t height, u_int32_t width, 
                                            uchar channels, int lastKernelIndex){
            if (lastKernelIndex >= 0){
                err = clSetKernelArg(kernel[0], 0, sizeof (cl_mem), &inputImage);
                err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg1");
                err = clSetKernelArg(kernel[0], 1, sizeof (cl_mem), &hsi1Converted);
                err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg1");
                err = clSetKernelArg(kernel[0], 2, sizeof (u_int32_t), &width);
                err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg1");
                err = clSetKernelArg(kernel[0], 3, sizeof (u_int32_t), &height);
                err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg1");
                err = clSetKernelArg(kernel[0], 4, sizeof (uchar), &channels);
                err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg1");
                
                if (lastKernelIndex >= 1){
                    err = clSetKernelArg(kernel[1], 0, sizeof (cl_mem), &inputImage);
                    err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg2");
                    err = clSetKernelArg(kernel[1], 1, sizeof (cl_mem), &hsi2Converted);
                    err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg2");
                    err = clSetKernelArg(kernel[1], 2, sizeof (u_int32_t), &width);
                    err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg2");
                    err = clSetKernelArg(kernel[1], 3, sizeof (u_int32_t), &height);
                    err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg2");
                    err = clSetKernelArg(kernel[1], 4, sizeof (uchar), &channels);
                    err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg2");
                }
            }
        }
        
        void OpenclTools::setKernelArgs2(u_int32_t height, u_int32_t width, unsigned char channels){
            err = clSetKernelArg(kernel[2], 0, sizeof (cl_mem), &hsi1Converted);
            err_check(err, "OpenclTools::setKernelArgs2 clSetKernelArg0");
            err = clSetKernelArg(kernel[2], 1, sizeof (cl_mem), &tsaiOutput);
            err_check(err, "OpenclTools::setKernelArgs2 clSetKernelArg1");
            err = clSetKernelArg(kernel[2], 2, sizeof (u_int32_t), &width);
            err_check(err, "OpenclTools::setKernelArgs2 clSetKernelArg2");
            err = clSetKernelArg(kernel[2], 3, sizeof (u_int32_t), &height);
            err_check(err, "OpenclTools::setKernelArgs2 clSetKernelArg3");
            err = clSetKernelArg(kernel[2], 4, sizeof (uchar), &channels);
            err_check(err, "OpenclTools::setKernelArgs2 clSetKernelArg4");
        }
        
        void OpenclTools::setKernelArgs3(u_int32_t height, u_int32_t width, unsigned char channels){
            err = clSetKernelArg(kernel[2], 0, sizeof (cl_mem), &hsi2Converted);
            err_check(err, "OpenclTools::setKernelArgs3 clSetKernelArg0");
            err = clSetKernelArg(kernel[2], 1, sizeof (cl_mem), &tsaiOutput);
            err_check(err, "OpenclTools::setKernelArgs3 clSetKernelArg1");
            err = clSetKernelArg(kernel[2], 2, sizeof (u_int32_t), &width);
            err_check(err, "OpenclTools::setKernelArgs3 clSetKernelArg2");
            err = clSetKernelArg(kernel[2], 3, sizeof (u_int32_t), &height);
            err_check(err, "OpenclTools::setKernelArgs3 clSetKernelArg3");
            err = clSetKernelArg(kernel[2], 4, sizeof (uchar), &channels);
            err_check(err, "OpenclTools::setKernelArgs3 clSetKernelArg4");
        }
        
        Mat* OpenclTools::processRGBImage(uchar* image, u_int32_t width, u_int32_t height, uchar channels) throw (SDException&) {
            if (image == 0) {
                return 0;
            }
            
            createBuffers(image, height, width, channels);            
            
            setKernelArgs1(height, width, channels, 1);            
            size_t local_ws = workGroupSize[0];
            size_t global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(command_queue, kernel[0], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "OpenclTools::processRGBImage clEnqueueNDRangeKernel0");
            clFlush(command_queue);
            clFinish(command_queue);
            
            setKernelArgs2(height, width, channels);
            local_ws = workGroupSize[2];
            global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(command_queue, kernel[2], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "OpenclTools::processRGBImage clEnqueueNDRangeKernel2");
            clReleaseMemObject(hsi1Converted);
            hsi1Converted = 0;
            ratios1 = 0;
            ratios1 = (uchar*)MemMenager::allocate<uchar>(width * height);
            if (ratios1 == 0) {
                SDException exc(SHADOW_NO_MEM, "OpenclTools::processRGBImage Calculate ratios1");
                throw exc;
            }
            err = clEnqueueReadBuffer(command_queue, tsaiOutput, CL_TRUE, 0, width * height, ratios1, 0, NULL, NULL);
            err_check(err, "OpenclTools::processRGBImage clEnqueueReadBuffer1");
            clFlush(command_queue);
            clFinish(command_queue);
            
            Mat* ratiosImage1 = OpenCV2Tools::get8bitImage(ratios1, height, width);            
            Mat* binarized1 = OpenCV2Tools::binarize(ratiosImage1);            
            MemMenager::delocate(ratios1);
            ratios1 = 0;
            delete ratiosImage1;

            local_ws = workGroupSize[1];
            global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(command_queue, kernel[1], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "OpenclTools::processRGBImage clEnqueueNDRangeKernel1");
            clReleaseMemObject(inputImage);
            inputImage = 0;
            clFlush(command_queue);
            clFinish(command_queue);
            
            setKernelArgs3(height, width, channels);
            local_ws = workGroupSize[2];
            global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(command_queue, kernel[2], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "OpenclTools::processRGBImage clEnqueueNDRangeKernel2");
            clReleaseMemObject(hsi2Converted);
            hsi2Converted = 0;
            ratios2 = 0;
            ratios2 = (uchar*)MemMenager::allocate<uchar>(width * height);
            if (ratios2 == 0) {
                SDException exc(SHADOW_NO_MEM, "OpenclTools::processRGBImage Calculate ratios2");
                throw exc;
            }
            err = clEnqueueReadBuffer(command_queue, tsaiOutput, CL_TRUE, 0, width * height, ratios2, 0, NULL, NULL);
            err_check(err, "OpenclTools::processRGBImage clEnqueueReadBuffer2");
            clFlush(command_queue);
            clFinish(command_queue);

            clReleaseMemObject(tsaiOutput);
            tsaiOutput = 0;            
            
            Mat* ratiosImage2 = OpenCV2Tools::get8bitImage(ratios2, height, width);            
            Mat* binarized2 = OpenCV2Tools::binarize(ratiosImage2);            
            MemMenager::delocate(ratios2);
            ratios2 = 0;
            delete ratiosImage2;
            
            Mat* processedImageMat = OpenCV2Tools::joinTwoOcl(*binarized1, *binarized2);
            delete binarized1;
            delete binarized2;                        
            return processedImageMat;             
        }
        
        string OpenclTools::getClassName(){
            return string("shadowdetection::opencl::OpenclTools");
        }

    }
}

#endif
