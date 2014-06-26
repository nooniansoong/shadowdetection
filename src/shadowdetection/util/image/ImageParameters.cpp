#include "ImageParameters.h"
#include "shadowdetection/opencv/OpenCV2Tools.h"

namespace shadowdetection{
    namespace util{
        namespace image{
            using namespace std;
            using namespace cv;
            using namespace shadowdetection::opencv2;
            
            float* ImageParameters::merge(float label, float** arrs, int arrsLen, int* arrSize, int& retSize) {
                retSize = 1;
                for (int i = 0; i < arrsLen; i++) {
                    retSize += arrSize[i];
                }
                float* retArr = new float[retSize];
                int counter = 0;
                retArr[counter] = label;
                for (int i = 0; i < arrsLen; i++) {
                    int size = arrSize[i];
                    float* arr = arrs[i];
                    for (int j = 0; j < size; j++) {
                        retArr[++counter] = arr[j];
                    }
                }
                return retArr;
            }
            
            float* ImageParameters::merge(float** arrs, int arrsLen, int* arrSize, int& retSize){
                retSize = 0;
                for (int i = 0; i < arrsLen; i++) {
                    retSize += arrSize[i];
                }
                float* retArr = new float[retSize];
                int counter = -1;
                for (int i = 0; i < arrsLen; i++) {
                    int size = arrSize[i];
                    float* arr = arrs[i];
                    for (int j = 0; j < size; j++) {
                        retArr[++counter] = arr[j];
                    }
                }
                return retArr;
            }
            
            float getLabel(uchar val) {
                if (val == 0)
                    return 0.f;
                else
                    return 1.f;
            }
            
            float** ImageParameters::getImageParameters(Mat& originalImage, Mat& maskImage, 
                                                        int& rowDimension,
                                                        int& pixelNum) throw (SDException&){
                if (originalImage.size().width != maskImage.size().width ||
                    originalImage.size().height != maskImage.size().height){
                    SDException exc(SHADOW_DIFFERENT_IMAGES_SIZES, "ImageParameters::getImageParameters, MaskImage");
                    throw exc;    
                }
                float** ret = 0;
                size_t maskStep = maskImage.step;
                int maskChan = maskImage.channels();
                if (maskChan > 1){
                    SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "ImageParameters::getImageParameters, MaskImage");
                    throw exc;
                }
                int height = originalImage.size().height;
                int width = originalImage.size().width;
                int noLabelDataRowDimension;
                int pixelCount;
                float** noLabel = getImageParameters(originalImage, noLabelDataRowDimension, pixelCount);
                if (noLabel != 0){
                    ret = new float*[width * height];
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            float label = getLabel(maskImage.data[i * maskStep + j * maskChan]);
                            int sizes[1];
                            sizes[0] = noLabelDataRowDimension;
                            float* procs[1];
                            procs[0] = noLabel[i * width + j];
                            int mergedSize;
                            float* merged = merge(label, procs, 1, sizes, mergedSize);
                            ret[i * width + j] = merged;
                            delete[] noLabel[i * width + j];                        
                            if (i == 0 && j == 0){
                                rowDimension = mergedSize;
                                pixelNum = width * height;
                            }
                        }
                    }
                    delete[] noLabel;
                }
                return ret;
            }
            
            float** ImageParameters::getImageParameters(cv::Mat& originalImage, int& rowDimension,
                                                        int& pixelNum) throw (SDException&){
                float** ret = 0;
                int height = originalImage.size().height;
                int width = originalImage.size().width;
                Mat* hsv = OpenCV2Tools::convertToHSV(&originalImage);
                size_t stepHsv = hsv->step;
                int chanHSV = hsv->channels();
                Mat* hls = OpenCV2Tools::convertToHLS(&originalImage);
                size_t stepHls = hls->step;
                int chanHLS = hls->channels();
                ret = new float*[width * height];
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        uchar hHSV = hsv->data[i * stepHsv + j * chanHSV + 0];
                        uchar sHSV = hsv->data[i * stepHsv + j * chanHSV + 1];
                        uchar vHSV = hsv->data[i * stepHsv + j * chanHSV + 2];

                        uchar hHLS = hls->data[i * stepHls + j * chanHLS + 0];
                        uchar lHLS = hls->data[i * stepHls + j * chanHLS + 1];
                        uchar sHLS = hls->data[i * stepHls + j * chanHLS + 1];

                        float* procs[SPACES_COUNT];
                        int size[SPACES_COUNT];
                        procs[0] = processHSV(hHSV, sHSV, vHSV, size[0]);
                        procs[1] = processHLS(hHLS, lHLS, sHLS, size[1]);

                        int mergedSize = 0;
                        float* merged = ImageParameters::merge(procs, SPACES_COUNT, size, mergedSize);

                        for (int k = 0; k < SPACES_COUNT; k++)
                            delete[] procs[k];
                        ret[i * width + j] = merged;
                        if (i == 0 && j == 0) {
                            rowDimension = mergedSize;
                            pixelNum = width * height;
                        }
                    }
                }
                return ret;
            }
            
            float* ImageParameters::processHSV(uchar H, uchar S, uchar V, int& size) {
                size = HSV_PARAMETERS;
                float* retArr = new float[size];
                //180 is max in opencv for H
                retArr[0] = (float) H / 180.f;
                retArr[0] = clamp(retArr[0], 0.f, 1.f);
                retArr[1] = (float) S / 255.f;
                retArr[1] = clamp(retArr[1], 0.f, 1.f);
                retArr[2] = (float) V / 255.f;
                retArr[2] = clamp(retArr[2], 0.f, 1.f);
                retArr[3] = (float) H / (float) (S + 1);
                retArr[3] /= 180.f;
                retArr[3] = clamp(retArr[3], 0.f, 1.f);
                retArr[4] = (float) H / (float) (V + 1);
                retArr[4] /= 180.f;
                retArr[4] = clamp(retArr[4], 0.f, 1.f);
                retArr[5] = (float) S / (float) (V + 1);
                retArr[5] /= 255.f;
                retArr[5] = clamp(retArr[5], 0.f, 1.f);
                return retArr;
            }

            float* ImageParameters::processHLS(uchar H, uchar L, uchar S, int& size) {
                size = HLS_PARAMETERS;
                float* retArr = new float[size];
                //180 is max in opencv for H
                retArr[0] = (float) H / 180.f;
                retArr[0] = clamp(retArr[0], 0.f, 1.f);
                retArr[1] = (float) L / 255.f;
                retArr[1] = clamp(retArr[1], 0.f, 1.f);
                retArr[2] = (float) S / 255.f;
                retArr[2] = clamp(retArr[2], 0.f, 1.f);
                retArr[3] = (float) H / (float) (L + 1);
                retArr[3] /= 180.f;
                retArr[3] = clamp(retArr[3], 0.f, 1.f);
                retArr[4] = (float) H / (float) (S + 1);
                retArr[4] /= 180.f;
                retArr[4] = clamp(retArr[4], 0.f, 1.f);
                retArr[5] = (float) L / (float) (S + 1);
                retArr[5] /= 255.f;
                retArr[5] = clamp(retArr[5], 0.f, 1.f);
                return retArr;
            }
            
        }
    }
}