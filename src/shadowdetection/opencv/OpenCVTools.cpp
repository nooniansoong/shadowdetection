#include "OpenCVTools.h"
#include "typedefs.h"

namespace shadowdetection {
    namespace opencv {
        
        using namespace std;
        using namespace cv::ocl;
        using namespace cv;
        
        unsigned char maxF(unsigned char a, unsigned char b, unsigned char c) {
            unsigned char max = a;
            if (b > max)
                max = b;
            if (c > max)
                max = c;

            return max;
        }

        unsigned char minF(unsigned char a, unsigned char b, unsigned char c) {
            unsigned char min = a;
            if (b < min)
                min = b;
            if (c < min)
                min = c;

            return min;
        }

        float radToDegrees(float radians) {
            const float PI_F = 3.14159265358979f;
            float oneRad = 180.f / PI_F;
            return radians * oneRad;
        }

        void OpenCvTools::RGBtoHSI_1(unsigned char r, unsigned char g, unsigned char b, unsigned int& h, unsigned char& s, unsigned char& i) {
            unsigned char min, max;// delta;

            min = minF(r, g, b);
            max = maxF(r, g, b);
            i = (unsigned char) (((float) (r + g + b)) / 3.f);

            //delta = max - min;
            float fS;
            float fH;

            float v1 = 0.5f * (2.f * r - g - b);
            float v2 = 0.5f * sqrt(3.f) * (g - b);

            fH = atan2(v2, v1);
            fH = radToDegrees(fH);

            if (fH < -360.f)
                fH = -360.f;

            if (fH < 0.)
                fH += 360.f;

            if (fH > 360.f)
                fH = 360.f;
            
            if (i != 0){
                fS = 1.f - ((float)min / (float) i);
            }
            else{
                fS = 0.f;
            }
            if (fS > 1.f) {
                fS = 1.f;
            }
            if (fS < 0.f){
                fS = 0.f;
            }    

            h = (unsigned int) fH;            
            s = (unsigned char) (fS * 255.f);            
        }

        void OpenCvTools::RGBtoHSI_2(unsigned char r, unsigned char g, unsigned char b, unsigned int& h, unsigned char& s, unsigned char& i) {
            float min, max, delta;

            min = minF(r, g, b);
            max = maxF(r, g, b);
            i = (unsigned char) (((float) (r + g + b)) / 3.f);

            delta = max - min;
            float fS;
            float fH;

            float v1 = -(((sqrt(6.f) * (float) r) / 6.f) +
                    ((sqrt(6.f) * (float) g) / 6.f) +
                    ((sqrt(6.f) * (float) b) / 3.f));

            float v2 = ((float) r / sqrt(6.f)) - ((2.f * (float) g) / sqrt(6.f));

            fH = atan2(v2, v1);
            fH = radToDegrees(fH);

            if (fH < -360.f)
                fH = -360.f;

            if (fH < 0.)
                fH += 360.f;

            if (fH > 360.f)
                fH = 360.f;
            if (i != 0){
                fS = 1.f - ((float)min / (float) i);
            }
            else{
                fS = 0.f;
            }            
            if (fS > 1.f) {
                fS = 1.f;
            }
            if (fS < 0.f){
                fS = 0.f;
            }

            h = (unsigned int) fH;
            s = (unsigned char) (fS * 255.f);
        }

        uint* OpenCvTools::convertImagetoHSI(IplImage* image, int& height, int& width, int& channels,
                void (*convertFunc)(unsigned char, unsigned char, unsigned char, unsigned int&, unsigned char&, unsigned char&)) {
            height = image->height;
            width = image->width;
            channels = image->nChannels;
            unsigned int* retArr = new unsigned int[height * width * channels];

            unsigned char* data = (unsigned char*) image->imageData;

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    unsigned char B = data[i * image->widthStep + j * channels + 0];
                    unsigned char G = data[i * image->widthStep + j * channels + 1];
                    unsigned char R = data[i * image->widthStep + j * channels + 2];

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

        unsigned char* OpenCvTools::convertImageToByteArray(IplImage* image) {
            if (image == 0) {
                return 0;
            }
            int height = image->height;
            int width = image->width;
            int channels = image->nChannels;
            unsigned char* retArr = new unsigned char[height * width * channels];
            unsigned char* data = (unsigned char*) image->imageData;

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    unsigned char B = data[i * image->widthStep + j * channels + 0];
                    unsigned char G = data[i * image->widthStep + j * channels + 1];
                    unsigned char R = data[i * image->widthStep + j * channels + 2];

                    int index = i * width * channels + j * channels;

                    retArr[index + 0] = R;
                    retArr[index + 1] = G;
                    retArr[index + 2] = B;
                }
            }

            return retArr;
        }

        IplImage* OpenCvTools::convertByteArrayToImage(unsigned char* arr, int width, int height, int channels) {
            if (arr == 0) {
                return 0;
            }

            IplImage* image = cvCreateImage(cvSize(width, height), 8, channels);
            if (image != 0) {
                unsigned char* data = (unsigned char*) image->imageData;

                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        int index = i * width * channels + j * channels;
                        unsigned char B = data[index + 2];
                        unsigned char G = data[index + 1];
                        unsigned char R = data[index];

                        data[i * image->widthStep + j * channels + 0] = B;
                        data[i * image->widthStep + j * channels + 1] = G;
                        data[i * image->widthStep + j * channels + 2] = R;
                    }
                }
            }

            return image;
        }

        uchar* OpenCvTools::simpleTsai(unsigned int* inputHSI, int height, int width, int channels) {
            unsigned char* retArr = new unsigned char[height * width];

            float maxVal = 360.f; // * 255.f;// + 1;
            float minVal = 0.f;
            float delta = maxVal - minVal;
            float segment = delta / 255.f;

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int index = i * width * channels + j * channels;

                    unsigned int H = inputHSI[index + 0];
                    unsigned int S = inputHSI[index + 1];
                    unsigned int I = inputHSI[index + 2];

                    float ratio = (float) (H) / ((float) (I) + 1.f);

                    ratio -= minVal;
                    ratio /= segment;
                    retArr[i * width + j] = (unsigned char) ratio;
                }
            }

            return retArr;
        }

        IplImage* OpenCvTools::get8bitImage(unsigned char* input, int height, int width) {
            IplImage* image = cvCreateImage(cvSize(width, height), 8, 1);
            int wStep = image->widthStep;
            unsigned char* data = (unsigned char*) image->imageData;

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    data[i * wStep + j] = input[i * width + j];
                }
            }

            return image;
        }

        IplImage* OpenCvTools::binarize(IplImage* input) {
            IplImage* image = cvCreateImage(cvSize(input->width, input->height), input->depth, input->nChannels);

            double thresh = cvThreshold(input, image, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            return image;
        }

        IplImage* OpenCvTools::joinTwo(IplImage* src1, IplImage* src2) {
            IplImage* image = cvCreateImage(cvSize(src1->width, src1->height), src1->depth, src1->nChannels);
            cvOr(src1, src2, image);
            return image;
        }               
        
//        IplImage* OpenCvTools::convertMatToIplImage(const Mat& matrix){
//            const int depth = matrix.depth(); 
//            IPL_DEPTH_8U;
//            int oldDepth;
//            switch (depth){
//                case CV_8U:
//                    oldDepth = IPL_DEPTH_8U;
//                    break;
//                case CV_8S:
//                    oldDepth = IPL_DEPTH_8S;
//                    break;
//                case CV_16U:
//                    oldDepth = IPL_DEPTH_16U;
//                    break;
//                case CV_16S:
//                    oldDepth = IPL_DEPTH_16S;
//                    break;
//                case CV_32S:
//                    oldDepth = IPL_DEPTH_32S;
//                    break;
//                case CV_32F:
//                    oldDepth = IPL_DEPTH_32F;
//                    break;
//                case CV_64F:
//                    return 0;
//                    break;
//                case CV_USRTYPE1:
//                    return 0;
//                    break;
//                default:
//                    oldDepth = IPL_DEPTH_8U;
//                    break;
//            }
//            int channels = matrix.channels();
//            IplImage* processedImage = cvCreateImage(cvSize(matrix.size().width, matrix.size().height), oldDepth, channels);
//            size_t ws = processedImage->widthStep;
//            size_t wsNew = matrix.step;
//            if (ws != wsNew){
//                cvReleaseImage(&processedImage);
//                cout << "ERROR" << endl;
//                return 0;
//            }
//            memcpy(processedImage->imageData, matrix.data, wsNew);
//            //processedImage->imageData = (char*)matrix.data;
//            return processedImage;
//        }

#ifdef _OPENCL 
        void OpenCvTools::initOpenCL(int pid, int device) throw (SDException&){
#ifdef _AMD
            int typeFlag = cv::ocl::CVCL_DEVICE_TYPE_CPU;
#else
            int typeFlag = cv::ocl::CVCL_DEVICE_TYPE_GPU;
#endif
            cv::ocl::PlatformsInfo platformsInfo;
            cv::ocl::getOpenCLPlatforms(platformsInfo);
            size_t size = platformsInfo.size();
            if (size <= 0 || pid >= size){
                SDException exc(SHADOW_NO_OPENCL_PLATFORM, "OpenCV Init OpenCL platforms");
                throw exc;
            }
            cv::ocl::DevicesInfo devicesInfo;            
            int devnums = cv::ocl::getOpenCLDevices(devicesInfo, typeFlag, (pid < 0) ? NULL : platformsInfo[pid]);
            if (device >= devnums){
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
        
        Mat* OpenCvTools::joinTwoOcl(const cv::Mat& src1, const cv::Mat& src2){            
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
