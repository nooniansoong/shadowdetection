#include <iostream>
#include "OpenCV2Tools.h"
#include "shadowdetection/util/Config.h"

namespace shadowdetection {
    namespace opencv2 {

        using namespace std;
        using namespace cv;
        using namespace cv::ocl;
        using namespace shadowdetection::util;
        
        unsigned int* OpenCV2Tools::convertImagetoHSI  (const Mat* image, int& height, int& width, int& channels,
                                             void (*convertFunc)(unsigned char, unsigned char, unsigned char, unsigned int&, unsigned char&, unsigned char&)){
            if (image == 0){
                return 0;
            }
            height = image->size().height;
            width = image->size().width;
            channels = image->channels();
            unsigned int* retArr = 0;
            retArr = (uint*)MemMenager::allocate<uint>(height * width * channels);
            if (retArr == 0){
                SDException exc(SHADOW_NO_MEM, "Convert to HSI");
            }
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
                uchar* retArr = 0;
                retArr = MemMenager::allocate<uchar>(height * width * channels);
                if (retArr == 0){
                    SDException exc(SHADOW_NO_MEM, "Convert image to array");
                    throw exc;
                }
                uchar* data = image->data;
                size_t step = image->step;
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        uchar B = data[i * step + j * channels + 0];
                        uchar G = data[i * step + j * channels + 1];
                        uchar R = data[i * step + j * channels + 2];
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

            Mat* image = 0;
            image = new(nothrow) Mat(height, width, CV_8UC3, arr);
            if (image == 0){
                SDException exc(SHADOW_NO_MEM, "Convert array to image");
                throw exc;
            }
            return image;
        }
        
        Mat* OpenCV2Tools::get8bitImage(unsigned char* input, int height, int width){
            Mat* image = 0;
            image = new(nothrow) Mat(height, width, CV_8U, input);
            if (image == 0){
                SDException exc(SHADOW_NO_MEM, "Get 8bit image");
                throw exc;
            }
            return image;
        }
        
        Mat* OpenCV2Tools::binarize(const Mat* input){
            Mat* image = 0;
            image = new(nothrow) Mat();
            if (image == 0){
                SDException exc(SHADOW_NO_MEM, "binarize");
                throw exc;
            }
            threshold(*input, *image, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            return image;
        }
        
        Mat* OpenCV2Tools::joinTwo(const Mat* src1, const Mat* src2){
            Mat* image = 0;
            image = new(nothrow) Mat();
            if (image == 0){
                SDException exc(SHADOW_NO_MEM, "Join two");
                throw exc;
            }
            bitwise_or(*src1, *src2, *image);
            return image;
        }
        
#ifdef _OPENCL 
        void OpenCV2Tools::initOpenCL(uint pid, uint device) throw (SDException&) {
#if defined _AMD || defined _MAC
            int typeFlag = cv::ocl::CVCL_DEVICE_TYPE_ALL;
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
#if defined _AMD || defined _MAC
            int devnums = cv::ocl::getOpenCLDevices(devicesInfo, typeFlag, NULL);
#else
            int devnums = cv::ocl::getOpenCLDevices(devicesInfo, typeFlag, (pid < 0) ? NULL : platformsInfo[pid]);
#endif
            //trust me, I know what I'm doing :)
            if ((int)device >= devnums) {
                SDException exc(SHADOW_NO_OPENCL_DEVICE, "OpenCV Init OpenCL devices");
                throw exc;
            }            
            cv::ocl::setDevice(devicesInfo[device]);
            string type = "DEVICE_OTHER";
            if (devicesInfo[device]->deviceType == CVCL_DEVICE_TYPE_CPU)
                type = "DEVICE_CPU";
            else if (devicesInfo[device]->deviceType == CVCL_DEVICE_TYPE_GPU)
                type = "DEVICE_GPU";
            cout << "Device type: " << type << endl;
            cout << "Platform name:" << devicesInfo[device]->platform->platformName << endl;
            cout << "Device name:" << devicesInfo[device]->deviceName << endl;
        }        

        Mat* OpenCV2Tools::joinTwoOcl(const cv::Mat& src1, const cv::Mat& src2) {
            oclMat oclSrc1(src1);
            oclMat oclSrc2(src2);
            oclMat res;
            ocl::bitwise_or(oclSrc1, oclSrc2, res);
            Mat* image = 0;
            image = new(nothrow) Mat(res);
            if (image == 0){
                SDException exc(SHADOW_NO_MEM, "Join two ocl");
                throw exc;
            }
            return image;
        }
#endif
        
        cv::Mat* OpenCV2Tools::convertToHSV(const cv::Mat* src) throw (SDException&){
            Mat* ret = 0;
#ifdef _OPENCL
            oclMat oclSrc(*src);
            oclMat res;
            ocl::cvtColor(oclSrc, res, CV_BGR2HSV);
            ret = new(nothrow) Mat(res);
            if (ret == 0){
                SDException exc(SHADOW_NO_MEM, "Convert to HSV");
                throw exc;
            }
#else
            ret = new(nothrow) Mat();
            if (ret == 0){
                SDException exc(SHADOW_NO_MEM, "Convert to HSV");
                throw exc;
            }
            cvtColor(*src, *ret, CV_BGR2HSV);
#endif
            return ret;
        }
        
        cv::Mat* OpenCV2Tools::convertToHLS(const cv::Mat* src) throw (SDException&){
            Mat* ret = 0;
#ifdef _OPENCL
            oclMat oclSrc(*src);
            oclMat res;
            ocl::cvtColor(oclSrc, res, CV_BGR2HLS);
            ret = new(nothrow) Mat(res);
            if (ret == 0){
                SDException exc(SHADOW_NO_MEM, "Convert to HSV");
                throw exc;
            }
#else
            ret = new(nothrow) Mat();
            if (ret == 0){
                SDException exc(SHADOW_NO_MEM, "Convert to HSV");
                throw exc;
            }
            cvtColor(*src, *ret, CV_BGR2HLS);
#endif
            return ret;
        }

    }
}