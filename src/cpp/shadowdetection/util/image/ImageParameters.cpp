#include "ImageParameters.h"
#include "shadowdetection/opencv/OpenCV2Tools.h"
#include "shadowdetection/util/MemMenager.h"
#include "shadowdetection/util/raii/RAIIS.h"
#include "shadowdetection/util/Config.h"

#define SPACES_COUNT 4
#define HSV_PARAMETERS 5
#define HLS_PARAMETERS 5
#define BGR_PARAMETERS 2
#define ROI_PARAMETERS 1;

namespace shadowdetection{
    namespace util{
        namespace image{
            using namespace std;
            using namespace cv;
            using namespace shadowdetection::opencv2;
            using namespace shadowdetection::util;
            using namespace shadowdetection::util::raii;
            
            ImageParameters::ImageParameters(){
                regionsAvgsSecondChannel = 0;
                numOfSegments = 1;
            }
            
            ImageParameters::~ImageParameters(){
                reset();
            }
            
            float* ImageParameters::merge(float label, const float** arrs, int arrsLen, int* arrSize, int& retSize) {
                retSize = 1;
                for (int i = 0; i < arrsLen; i++) {
                    retSize += arrSize[i];
                }
                float* retArr = MemMenager::allocate<float>(retSize);
                if (retArr){
                    int counter = 0;
                    retArr[counter] = label;
                    for (int i = 0; i < arrsLen; i++) {
                        int size = arrSize[i];
                        const float* arr = arrs[i];
                        for (int j = 0; j < size; j++) {
                            retArr[++counter] = arr[j];
                        }
                    }
                }
                return retArr;
            }
            
            float* ImageParameters::merge(float** arrs, int arrsLen, int* arrSize, int& retSize){
                retSize = 0;
                for (int i = 0; i < arrsLen; i++) {
                    retSize += arrSize[i];
                }
                float* retArr = MemMenager::allocate<float>(retSize);
                if (retArr){
                    int counter = -1;
                    for (int i = 0; i < arrsLen; i++) {
                        int size = arrSize[i];
                        float* arr = arrs[i];
                        for (int j = 0; j < size; j++) {
                            retArr[++counter] = arr[j];
                        }
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
            
            Matrix<float>* ImageParameters::getImageParameters(const Mat& originalImage, const Mat& maskImage, 
                                                        int& rowDimension, int& pixelNum) throw (SDException&){
                if (originalImage.data == 0 || maskImage.data == 0)
                    return 0;
                if (originalImage.size().width != maskImage.size().width ||
                    originalImage.size().height != maskImage.size().height){
                    SDException exc(SHADOW_DIFFERENT_IMAGES_SIZES, "ImageParameters::getImageParameters, MaskImage");
                    throw exc;    
                }
                Matrix<float>* ret = 0;
                PointerRaii< Matrix<float> > retRaii;
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
                const Matrix<float>* noLabel = getImageParameters(originalImage, noLabelDataRowDimension, pixelCount);
                if (noLabel != 0){
                    PointerRaii< const Matrix<float> > noLabelRaii(noLabel);
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            float label = getLabel(maskImage.data[i * maskStep + j * maskChan]);
                            int sizes[1];
                            sizes[0] = noLabelDataRowDimension;
                            const float* procs[1];
                            procs[0] = (*noLabel)[i * width + j];
                            int mergedSize;
                            float* merged = merge(label, procs, 1, sizes, mergedSize);
                            if (merged == 0){                                
                                return 0;
                            }
                            VectorRaii vraii(merged);
                            if (ret == 0){
                                ret = new Matrix<float>(mergedSize, width * height);
                                retRaii.setPointer(ret);
                            }
                            
                            (*ret)[i * width + j] = merged;                                                    
                            if (i == 0 && j == 0){
                                rowDimension = mergedSize;
                                pixelNum = width * height;
                            }
                        }
                    }                    
                }
                else{
                    return 0;
                }
                retRaii.deactivate();
                return ret;
            }
            
            Matrix<float>* ImageParameters::getImageParameters(const Mat& originalImage, int& rowDimension,
                                                        int& pixelNum) throw (SDException&){
                if (originalImage.data == 0)
                    return 0;
                Matrix<float>* ret = 0;
                PointerRaii< Matrix<float> > retRaii;
                int height = originalImage.size().height;
                int width = originalImage.size().width;
                Mat* hsv = OpenCV2Tools::convertToHSV(&originalImage);
                if (hsv->data == 0){
                    return 0;
                }
                ImageNewRaii imhsvRaii(hsv);
                size_t stepHsv = hsv->step;
                int chanHSV = hsv->channels();
                Mat* hls = OpenCV2Tools::convertToHLS(&originalImage);
                if (hls->data == 0){
                    return 0;
                }
                ImageNewRaii imhlsRaii(hls);
                size_t stepHls = hls->step;
                int chanHLS = hls->channels();                
                
                size_t stepBgr = originalImage.step;
                int chanBGR = originalImage.channels();
                
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        uchar hHSV = hsv->data[i * stepHsv + j * chanHSV + 0];
                        uchar sHSV = hsv->data[i * stepHsv + j * chanHSV + 1];
                        uchar vHSV = hsv->data[i * stepHsv + j * chanHSV + 2];

                        uchar hHLS = hls->data[i * stepHls + j * chanHLS + 0];
                        uchar lHLS = hls->data[i * stepHls + j * chanHLS + 1];
                        uchar sHLS = hls->data[i * stepHls + j * chanHLS + 2];

                        float* procs[SPACES_COUNT];
                        int size[SPACES_COUNT];
                        procs[0] = processHSV(hHSV, sHSV, vHSV, size[0]);
                        if (procs[0] == 0){
                            return 0;
                        }
                        VectorRaii vraiiProc0(procs[0]);
                        procs[1] = processHLS(hHLS, lHLS, sHLS, size[1]);
                        if (procs[1] == 0){
                            return 0;
                        }
                        VectorRaii vraiiProc1(procs[1]);
                        uchar B = originalImage.data[i * stepBgr + j * chanBGR + 0];
                        uchar G = originalImage.data[i * stepBgr + j * chanBGR + 1];
                        uchar R = originalImage.data[i * stepBgr + j * chanBGR + 2];
                        procs[2] = processBGR(B, G, R, size[2]);
                        if (procs[2] == 0){
                            return 0;
                        }
                        VectorRaii vraiiProc2(procs[2]);
                        
                        KeyVal<uint> location((uint)j, (uint)i);
                        procs[3] = processROI(location, hls, size[3], 1);
                        if (procs[3] == 0){
                            return 0;
                        }
                        VectorRaii vraiiProcs3(procs[3]);
                        
                        int mergedSize = 0;
                        float* merged = ImageParameters::merge(procs, SPACES_COUNT, size, mergedSize);
                        if (merged == 0){                            
                            return 0;
                        }
                        VectorRaii vraii(merged);
                        if (ret == 0){
                            ret = new Matrix<float>(mergedSize, width * height);
                            retRaii.setPointer(ret);
                        }
                                                
                        (*ret)[i * width + j] = merged;                        
                        if (i == 0 && j == 0) {
                            rowDimension = mergedSize;
                            pixelNum = width * height;
                        }
                    }
                }
                retRaii.deactivate();
                return ret;
            }
            
            float* ImageParameters::processHSV(uchar H, uchar S, uchar V, int& size) {
                size = HSV_PARAMETERS;
                float* retArr = MemMenager::allocate<float>(size);
                if (retArr != 0){
                //180 is max in opencv for H
//                    retArr[0] = (float) H / 180.f;
//                    retArr[0] = clamp(retArr[0], 0.f, 1.f);
                    retArr[0] = (float) S / 255.f;
                    retArr[0] = clamp<float>(retArr[0], 0.f, 1.f);
                    retArr[1] = (float) V / 255.f;
                    retArr[1] = clamp<float>(retArr[1], 0.f, 1.f);
                    retArr[2] = (float) H / (float) (S + 1);
                    retArr[2] /= 180.f;
                    retArr[2] = clamp<float>(retArr[2], 0.f, 1.f);
                    retArr[3] = (float) H / (float) (V + 1);
                    retArr[3] /= 180.f;
                    retArr[3] = clamp<float>(retArr[3], 0.f, 1.f);
                    retArr[4] = (float) S / (float) (V + 1);
                    retArr[4] /= 255.f;
                    retArr[4] = clamp<float>(retArr[4], 0.f, 1.f);
                }
                return retArr;
            }

            float* ImageParameters::processHLS(uchar H, uchar L, uchar S, int& size) {
                size = HLS_PARAMETERS;
                float* retArr = MemMenager::allocate<float>(size);
                if (retArr != 0){
                //180 is max in opencv for H
//                    retArr[0] = (float) H / 180.f;
//                    retArr[0] = clamp(retArr[0], 0.f, 1.f);
                    retArr[0] = (float) L / 255.f;
                    retArr[0] = clamp<float>(retArr[0], 0.f, 1.f);
                    retArr[1] = (float) S / 255.f;
                    retArr[1] = clamp<float>(retArr[1], 0.f, 1.f);
                    retArr[2] = (float) H / (float) (L + 1);
                    retArr[2] /= 180.f;
                    retArr[2] = clamp<float>(retArr[2], 0.f, 1.f);
                    retArr[3] = (float) H / (float) (S + 1);
                    retArr[3] /= 180.f;
                    retArr[3] = clamp<float>(retArr[3], 0.f, 1.f);
                    retArr[4] = (float) L / (float) (S + 1);
                    retArr[4] /= 255.f;
                    retArr[4] = clamp<float>(retArr[4], 0.f, 1.f);
                }
                return retArr;                
            }
            
            float* ImageParameters::processBGR(uchar B, uchar G, uchar R, int& size){
                size = BGR_PARAMETERS;
                float* retArr = MemMenager::allocate<float>(size);
                if (retArr != 0){
                    retArr[0] = (float)B / 255.f;
                    retArr[0] = clamp<float>(retArr[0], 0.f, 1.f);
                    retArr[1] = (float)(G + R) / (255.f + 255.f);
                    retArr[1] = clamp<float>(retArr[1], 0.f, 1.f);
                }
                return retArr;
            }
            
            Matrix<float>* ImageParameters::getAvgChannelValForRegions(const Mat* originalImage, uchar channelIndex){                                                
                regionsAvgsSecondChannel = new Matrix<float>(numOfSegments, numOfSegments);
                segmentWidth = (float)originalImage->cols / (float)numOfSegments;
                segmentHeight = (float)originalImage->rows / (float)numOfSegments;
                for (int i = 0; i < numOfSegments; i++){
                    for (int j = 0; j < numOfSegments; j++){                
                        float xStart = j * segmentWidth;
                        float yStart = i * segmentHeight;
                        KeyVal<uint> location((uint)xStart, (uint)yStart);
                        Mat* roi = OpenCV2Tools::getImageROI(originalImage, segmentWidth, 
                                                            segmentHeight, location);                                                
                        float avg = OpenCV2Tools::getAvgChannelValue(roi, channelIndex);
                        (*regionsAvgsSecondChannel)[i][j] = avg;
                        delete roi;
                    }
                }
                return regionsAvgsSecondChannel;
            }
            
            float* ImageParameters::processROI( KeyVal<uint> location, const Mat* originalImage, 
                                                int& size, uchar channelIndex) throw (SDException&){
                if (originalImage == 0 || originalImage->data == 0){
                    SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "ImageParameters::processROI");
                    throw (exc);
                }
                if (regionsAvgsSecondChannel == 0){
                    numOfSegments = 16;
                    Config* config = Config::getInstancePtr();
                    string numSegmentsStr = config->getPropertyValue("settings.Parameters.numSegments");
                    if (numSegmentsStr != ""){
                        numOfSegments = atoi(numSegmentsStr.c_str());
                    }
                    getAvgChannelValForRegions(originalImage, channelIndex);
                }                 
                int index = location.getVal() * originalImage->step + location.getKey() * originalImage->channels() + channelIndex;
                float value = (float)originalImage->data[index];
                
                int yAvgIndex = location.getVal() / segmentHeight;
                int xAvgIndex = location.getKey() / segmentWidth;
                if (yAvgIndex > 0 && xAvgIndex > 0){
                    int a = 0;
                    ++a;
                }
                    
                float avg = (*regionsAvgsSecondChannel)[yAvgIndex][xAvgIndex];
                float proportion = value / (avg + 1.f);                
                proportion = atan(proportion / 3.f);
                proportion = (proportion + M_PI_2) * M_1_PI;
                proportion = clamp<float>(proportion, 0.f, 1.f);
                float* ret = MemMenager::allocate<float>(1);
                ret[0] = proportion;
                size = ROI_PARAMETERS;
                return ret;
            }
            
            void ImageParameters::reset(){
                if (regionsAvgsSecondChannel)
                    delete regionsAvgsSecondChannel;
                regionsAvgsSecondChannel = 0;
                numOfSegments = 1;
            }
            
        }
    }
}