
#include "core/opencl/OpenClToolsBase.h"
#include "OpenCLImageParameters.h"
#include "core/util/Matrix.h"
#include "core/opencv/OpenCV2Tools.h"

#ifdef _OPENCL

namespace shadowdetection {
    namespace opencl {
        
        using namespace std;
        using namespace cv;
        using namespace core::util;
        using namespace core::opencv2;
        
        string OpenCLImageParameters::getClassName(){
            return string("shadowdetection::opencl::OpenCLImageParameters");
        }
        
        OpenCLImageParameters::OpenCLImageParameters() : Singleton<OpenCLImageParameters>(){
            initVars();
        }
        
        OpenCLImageParameters::~OpenCLImageParameters(){
            
        }

        void OpenCLImageParameters::initVars(){
            OpenClBase::initVars();
            
            initWorkVars();
        }
        
        void OpenCLImageParameters::initWorkVars(){
            parametersMem = 0;
            originalImageBuffer = 0;
            hsvImageBuffer = 0;
            hlsImageBuffer = 0;
        }
        
        void OpenCLImageParameters::cleanUp(){
            OpenClBase::cleanUp();
            cleanWorkPart();
            initVars();
        }
        
        void OpenCLImageParameters::cleanWorkPart(){
            if (parametersMem)
                clReleaseMemObject(parametersMem);
            if (originalImageBuffer)
                clReleaseMemObject(originalImageBuffer);
            if (hsvImageBuffer)
                clReleaseMemObject(hsvImageBuffer);
            if (hlsImageBuffer)
                clReleaseMemObject(hlsImageBuffer);
            initWorkVars();
        }
        
        Matrix<float>* OpenCLImageParameters::getImageParameters(const Mat* originalImage, const Mat* hsvImage,
                                                                const Mat* hlsImage, const int& parameterCount){
            int imageWidth = originalImage->cols;
            int imageHeight = originalImage->rows;            
            int numOfPixels = imageWidth * imageHeight;
            Matrix<float>* retMat = new Matrix<float>(parameterCount, numOfPixels);
            createBuffers(numOfPixels, parameterCount, originalImage,
                            hsvImage, hlsImage);
            setKernelArgs(parameterCount, numOfPixels);
            size_t local_ws = workGroupSize[0];
            size_t global_ws = shrRoundUp(local_ws, numOfPixels);
            err = clEnqueueNDRangeKernel(command_queue, kernel[0], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "OpenCLImageParameters::getImageParameters clEnqueueNDRangeKernel0");
            float* parameters = retMat->getVec();
            err = clEnqueueReadBuffer(  command_queue, parametersMem, CL_TRUE, 0, 
                                        numOfPixels * parameterCount * sizeof(cl_float), 
                                        parameters, 0, NULL, NULL);
            clFlush(command_queue);
            clFinish(command_queue);
            err_check(err, "OpenclTools::processRGBImage clEnqueueReadBuffer1");
            return retMat;
        }
        
        void OpenCLImageParameters::createBuffers(const int& numOfPixels, const int parameterCount,
                                                    const Mat* originalImage, const Mat* hsvImage, 
                                                    const Mat* hlsImage){
            cl_device_type type;
            err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof (cl_device_type), &type, 0);
            err_check(err, "OpenCLImageParameters::createBuffers clGetDeviceInfo");
            int flag1, flag2;
            if (type == CL_DEVICE_TYPE_GPU) {
                flag1 = CL_MEM_WRITE_ONLY;
                flag2 = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
            } else if (type == CL_DEVICE_TYPE_CPU) {
                flag1 = CL_MEM_WRITE_ONLY;
                flag2 = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
            } else {
                SDException exc(SHADOW_NOT_SUPPORTED_DEVICE, "Init buffers, currently not supported device");
                throw exc;
            }
                        
            int imageChannels = originalImage->channels();
            
            size_t numOfElements = numOfPixels * parameterCount;
            size_t size = sizeof(cl_float) * numOfElements;
            parametersMem = clCreateBuffer(context, flag1, size, 0, &err);
            err_check(err, "OpenCLToolsTrain::createBuffersSVM clCreateBuffer parametersMem");
            
            size = numOfPixels * imageChannels;
            
            uchar* orIm = OpenCV2Tools::convertImageToByteArray(originalImage, false);            
            originalImageBuffer = clCreateBuffer(context, flag2, size, orIm, &err);
            err_check(err, "OpenCLToolsTrain::createBuffersSVM clCreateBuffer originalImageBuffer");
            
            uchar* hsvIm = OpenCV2Tools::convertImageToByteArray(hsvImage, false);
            hsvImageBuffer = clCreateBuffer(context, flag2, size, hsvIm, &err);
            err_check(err, "OpenCLToolsTrain::createBuffersSVM clCreateBuffer hsvImageBuffer");
            
            uchar* hlsIm = OpenCV2Tools::convertImageToByteArray(hlsImage, false);
            hlsImageBuffer = clCreateBuffer(context, flag2, size, hlsIm, &err);
            err_check(err, "OpenCLToolsTrain::createBuffersSVM clCreateBuffer hlsImageBuffer");
        }
        
        void OpenCLImageParameters::setKernelArgs(  const cl_uint& numOfParameters, 
                                                    const cl_uint& numOfPixels){
            err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &parametersMem);
            err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg parametersMem");
            err = clSetKernelArg(kernel[0], 1, sizeof(cl_uint), &numOfParameters);
            err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg numOfParameters");
            err = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &originalImageBuffer);
            err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg originalImageBuffer");
            err = clSetKernelArg(kernel[0], 3, sizeof(cl_mem), &hsvImageBuffer);
            err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg hsvImageBuffer");
            err = clSetKernelArg(kernel[0], 4, sizeof(cl_mem), &hlsImageBuffer);
            err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg hlsImageBuffer");
            err = clSetKernelArg(kernel[0], 5, sizeof(cl_uint), &numOfPixels);
            err_check(err, "OpenclTools::setKernelArgs1 clSetKernelArg numOfPixels");
        }
        
    }
}

#endif
