#include "OpenCVTools.h"
#include "typedefs.h"
#include "shadowdetection/util/MemMenager.h"

namespace shadowdetection {
    namespace opencv {
        
        using namespace std;
        using namespace shadowdetection::util;

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

        unsigned int* OpenCvTools::convertImagetoHSI(IplImage* image, int& height, int& width, int& channels,
                void (*convertFunc)(unsigned char, unsigned char, unsigned char, unsigned int&, unsigned char&, unsigned char&)) {
            height = image->height;
            width = image->width;
            channels = image->nChannels;
            uint* retArr = 0;
            retArr = MemMenager::allocate<uint>(height * width * channels);
            if (retArr == 0){
                SDException exc(SHADOW_NO_MEM, "convertImagetoHSI");
                throw exc;
            }
            uchar* data = (uchar*) image->imageData;

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    uchar B = data[i * image->widthStep + j * channels + 0];
                    uchar G = data[i * image->widthStep + j * channels + 1];
                    uchar R = data[i * image->widthStep + j * channels + 2];

                    uint H;
                    uchar S;
                    uchar I;

                    convertFunc(R, G, B, H, S, I);
                    int index = i * width * channels + j * channels;

                    retArr[index + 0] = H;
                    retArr[index + 1] = (uint) S;
                    retArr[index + 2] = (uint) I;
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
            uchar* retArr = 0;
            retArr = MemMenager::allocate<uchar>(height * width * channels);
            if (retArr == 0){
                SDException exc(SHADOW_NO_MEM, "convertImageToByteArray");
                throw exc;
            }
            unsigned char* data = (unsigned char*) image->imageData;

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    uchar B = data[i * image->widthStep + j * channels + 0];
                    uchar G = data[i * image->widthStep + j * channels + 1];
                    uchar R = data[i * image->widthStep + j * channels + 2];

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
            uchar* retArr = 0;
            retArr = MemMenager::allocate<uchar>(height * width);
            if (retArr == 0){
                SDException exc(SHADOW_NO_MEM, "simpleTsai");
                throw exc;
            }

            float maxVal = 360.f; // * 255.f;// + 1;
            float minVal = 0.f;
            float delta = maxVal - minVal;
            float segment = delta / 255.f;

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int index = i * width * channels + j * channels;

                    uint H = inputHSI[index + 0];
                    uint S = inputHSI[index + 1];
                    uint I = inputHSI[index + 2];

                    float ratio = (float) (H) / ((float) (I) + 1.f);

                    ratio -= minVal;
                    ratio /= segment;
                    retArr[i * width + j] = (uchar)ratio;
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
        
    }
}
