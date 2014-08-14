#include <iostream>
#include "OpenCV2Tools.h"
#include "core/util/Config.h"

namespace core{
    namespace opencv2 {

        using namespace std;
        using namespace cv;
        using namespace cv::ocl;
        using namespace core::util;
        
        unsigned int* OpenCV2Tools::convertImagetoHSI  (const Mat* image, int& height, int& width, int& channels,
                                             void (*convertFunc)(unsigned char, unsigned char, unsigned char, unsigned int&, unsigned char&, unsigned char&)){
            if (image == 0 || image->data == 0){
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
            if (image == 0 || image->data == 0) {
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
            if (input == 0) {
                return 0;
            }
            Mat* image = 0;
            image = new(nothrow) Mat(height, width, CV_8U, input);
            if (image == 0){
                SDException exc(SHADOW_NO_MEM, "Get 8bit image");
                throw exc;
            }
            return image;
        }
        
        Mat* OpenCV2Tools::get8bitImage(int height, int width){
            Mat* image = 0;
            image = new(nothrow) Mat(height, width, CV_8U);
            if (image == 0){
                SDException exc(SHADOW_NO_MEM, "Get 8bit image");
                throw exc;
            }
            return image;
        }
        
        Mat* OpenCV2Tools::binarize(const Mat* input){
            if (input == 0 || input->data == 0) {
                return 0;
            }
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
            if (src1 == 0 || src2 == 0 || src1->data == 0 || src2->data == 0) {
                return 0;
            }
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

        Mat* OpenCV2Tools::joinTwoOcl(const Mat& src1, const Mat& src2) {
            if (src1.data == 0 || src2.data == 0){
                return 0;
            }
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
        
        Mat* OpenCV2Tools::convertToHSV(const Mat* src) throw (SDException&){
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
        
        Mat* OpenCV2Tools::convertToHLS(const Mat* src) throw (SDException&){
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
        
        /**
         * don't want this parallel
         * @param src
         * @param roiWidth
         * @param location
         * @return 
         */
        Mat* OpenCV2Tools::getImageROI( const Mat* src, uint roiWidth, uint roiHeight, 
                                        const KeyVal<uint>& location) throw (SDException&){
            if (src == 0 || src->data == 0){
                return 0;
            }
            int startX = location.getKey();
            int startY = location.getVal();
                         
            if (startX < 0)
                startX = 0;
            if (startX > src->cols - 1)
                return 0;
            
            if (startY < 0)
                startY = 0;
            if (startY > src->rows - 1)
                return 0;
            
            int endX = startX + roiWidth;
            if (endX < 0)
                return 0;
            endX = clamp<int>(endX, 0, src->cols - 1U);
            
            int endY = startY + roiHeight;
            if (endY < 0)
                return 0;
            endY = clamp<int>(endY, 0, src->rows - 1U);
            int diffX = endX - startX;
            int diffY = endY - startY;
            if (diffX <= 0 || diffY <= 0)
                return 0;
            Mat* retMat = new Mat(*src, Rect(startX, startY, diffX, diffY));
            return retMat;
        }
        
        /**
         * this also don't want parallel
         * @param src
         * @param channelIndex
         * @return 
         */
        float OpenCV2Tools::getAvgChannelValue( const Mat* src, 
                                                uchar channelIndex) throw (SDException&){
            if (src == 0 || src->data == 0 || (src->channels() - 1 < channelIndex)){
                SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "OpenCV2Tools::getAvgChannelValue");
                throw exc;
            }
            size_t step = src->step;
            uint64 val = 0;
            int channels = src->channels();            
            uint rows = src->rows;
            uint cols = src->cols;
            for (uint i = 0; i < rows; i++){
                for (uint j = 0; j < cols; j++){
                    int index = (i * step) + (j * channels) + channelIndex;
                    uchar chnVal = src->data[index];
                    val += chnVal;
                }
            }
            uint num = rows * cols;
            float retVal = (float)val / (float)num;
            return retVal;
        }
        
        uchar OpenCV2Tools::getChannelValue(const Mat& image, uint x, uint y, 
                                            uchar channelIndex) throw (SDException&){
            if (image.data == 0){
                SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "OpenCV2Tools::getChannelValue");
                throw exc;
            }
            size_t step = image.step;
            int channels = image.channels();
            if (channelIndex > channels - 1){
                SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "OpenCV2Tools::getChannelValue check index");
                throw exc;
            }
            if (x >= (uint)image.cols){
                SDException exc(SHADOW_OUT_OF_BOUNDS, "OpenCV2Tools::getChannelValue x");
                throw exc;
            }
            if (y >= (uint)image.rows){
                SDException exc(SHADOW_OUT_OF_BOUNDS, "OpenCV2Tools::getChannelValue y");
                throw exc;
            }
            
            int index = y * step + x * channels + channelIndex;
            return image.data[index];
        }
        
        uchar OpenCV2Tools::getChannelValue(const Mat& image, KeyVal<uint> location, 
                                            uchar channelIndex) throw (SDException&){
            uint x = location.getKey();
            uint y = location.getVal();
            return getChannelValue(image, x, y, channelIndex);
        }
        
        void OpenCV2Tools::setChannelValue( Mat& image, KeyVal<uint> location, 
                                            uchar channelIndex, uchar newValue) throw (SDException&){
            if (image.data == 0){
                SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "OpenCV2Tools::setChannelValue");
                throw exc;
            }
            size_t step = image.step;
            int channels = image.channels();
            if (channelIndex > channels - 1){
                SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "OpenCV2Tools::setChannelValue check index");
                throw exc;
            }
            uint x = location.getKey();
            uint y = location.getVal();
            if (x >= (uint)image.cols){
                SDException exc(SHADOW_OUT_OF_BOUNDS, "OpenCV2Tools::getChannelValue x");
                throw exc;
            }
            if (y >= (uint)image.rows){
                SDException exc(SHADOW_OUT_OF_BOUNDS, "OpenCV2Tools::getChannelValue y");
                throw exc;
            }
            int index = y * step + x * channels + channelIndex;
            image.data[index] = newValue;
        }

    }
}