#include <iostream>
#include "OpenCV2Tools.h"
#include "shadowdetection/util/Config.h"

namespace shadowdetection {
    namespace opencv2 {

        using namespace std;
        using namespace cv;
        using namespace cv::ocl;
        using namespace shadowdetection::util;
        
        uint* OpenCV2Tools::convertImagetoHSI  (const Mat* image, int& height, int& width, int& channels,
                                             void (*convertFunc)(unsigned char, unsigned char, unsigned char, unsigned int&, unsigned char&, unsigned char&)){
            if (image == 0){
                return 0;
            }
            height = image->size().height;
            width = image->size().width;
            channels = image->channels();
            unsigned int* retArr = new unsigned int[height * width * channels];
            unsigned char* data = (unsigned char*) image->data;
            size_t step = image->step;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    unsigned char B = data[i * step + j * channels + 0];
                    unsigned char G = data[i * step + j * channels + 1];
                    unsigned char R = data[i * step + j * channels + 2];

                    unsigned int H;
                    unsigned char S;
                    unsigned char I;

                    convertFunc(R, G, B, H, S, I);
                    int index = i * width * channels + j * channels;

                    retArr[index + 0] = H;
                    retArr[index + 1] = (unsigned int) S;
                    retArr[index + 2] = (unsigned int) I;
                }
            }
            return retArr;
        }
                
        unsigned char* OpenCV2Tools::convertImageToByteArray(const Mat* image, bool copy){
            if (image == 0) {
                return 0;
            }
            if (copy == true){
                int height = image->size().height;
                int width = image->size().width;
                int channels = image->channels();
                unsigned char* retArr = new unsigned char[height * width * channels];
                unsigned char* data = image->data;
                size_t step = image->step;
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        unsigned char B = data[i * step + j * channels + 0];
                        unsigned char G = data[i * step + j * channels + 1];
                        unsigned char R = data[i * step + j * channels + 2];
                        int index = i * width * channels + j * channels;
                        retArr[index + 0] = R;
                        retArr[index + 1] = G;
                        retArr[index + 2] = B;
                    }
                }
                return retArr;
            }
            else{
                return image->data;
            }
        }
        
        Mat* OpenCV2Tools::convertByteArrayToImage(unsigned char* arr, int width, int height, int channels){
            if (arr == 0) {
                return 0;
            }

            Mat* image = new Mat(height, width, CV_8UC3, arr);
//            if (image != 0) {
//                unsigned char* data = (unsigned char*) image->data;
//                size_t step = image->step;
//                for (int i = 0; i < height; i++) {
//                    for (int j = 0; j < width; j++) {
//                        int index = i * width * channels + j * channels;
//                        unsigned char B = data[index + 2];
//                        unsigned char G = data[index + 1];
//                        unsigned char R = data[index];
//
//                        data[i * step + j * channels + 0] = B;
//                        data[i * step + j * channels + 1] = G;
//                        data[i * step + j * channels + 2] = R;
//                    }
//                }
//            }
            return image;
        }
        
        Mat* OpenCV2Tools::get8bitImage(unsigned char* input, int height, int width){
            Mat* image = new Mat(height, width, CV_8U, input);
            return image;
        }
        
        Mat* OpenCV2Tools::binarize(const Mat* input){
            Mat* image = new Mat();
            double thresh = threshold(*input, *image, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            return image;
        }
        
        Mat* OpenCV2Tools::joinTwo(const Mat* src1, const Mat* src2){
            Mat* image = new Mat();
            bitwise_or(*src1, *src2, *image);
            return image;
        }
        
#ifdef _OPENCL 
        void OpenCV2Tools::initOpenCL(int pid, int device) throw (SDException&) {
#ifdef _AMD
            int typeFlag = cv::ocl::CVCL_DEVICE_TYPE_CPU;
#elif defined _MAC
            int typeFlag;
            string useGPUStr = Config::getInstancePtr()->getPropertyValue("settings.openCL.mac.useGPU");
            if (useGPUStr.compare("true") == 0)
                typeFlag = cv::ocl::CVCL_DEVICE_TYPE_GPU;
            else
                typeFlag = cv::ocl::CVCL_DEVICE_TYPE_CPU;
#else
            int typeFlag = cv::ocl::CVCL_DEVICE_TYPE_GPU;
#endif
            cv::ocl::PlatformsInfo platformsInfo;
            cv::ocl::getOpenCLPlatforms(platformsInfo);
            size_t size = platformsInfo.size();
            if (size <= 0 || pid >= size) {
                SDException exc(SHADOW_NO_OPENCL_PLATFORM, "OpenCV Init OpenCL platforms");
                throw exc;
            }
            cv::ocl::DevicesInfo devicesInfo;
            int devnums = cv::ocl::getOpenCLDevices(devicesInfo, typeFlag, (pid < 0) ? NULL : platformsInfo[pid]);
            if (device >= devnums) {
                SDException exc(SHADOW_NO_OPENCL_DEVICE, "OpenCV Init OpenCL devices");
                throw exc;
            }
            cv::ocl::setDevice(devicesInfo[device]);
            cout << "Device type: GPU" << endl;
            cout << "Platform name:" << devicesInfo[device]->platform->platformName << endl;
            cout << "Device name:" << devicesInfo[device]->deviceName << endl;
        }

        //        Mat* OpenCvTools::binarizeOcl(const Mat& image){
        //            cv::ocl::oclMat oclMatrix(image);
        //            cv::ocl::oclMat oclThreshed;
        //            ocl::threshold(oclMatrix, oclThreshed, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);            
        //            Mat* retMat = new Mat(oclThreshed);
        //            return retMat;
        //        }

        Mat* OpenCV2Tools::joinTwoOcl(const cv::Mat& src1, const cv::Mat& src2) {
            oclMat oclSrc1(src1);
            oclMat oclSrc2(src2);
            oclMat res;
            ocl::bitwise_or(oclSrc1, oclSrc2, res);
            Mat* image = new Mat(res);
            return image;
        }
#endif

    }
}