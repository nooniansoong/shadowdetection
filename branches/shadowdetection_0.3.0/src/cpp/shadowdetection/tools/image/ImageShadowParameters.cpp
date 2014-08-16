#include "ImageShadowParameters.h"
#include "core/opencv/OpenCV2Tools.h"
#include "core/util/MemMenager.h"
#include "core/util/raii/RAIIS.h"
#include "core/util/Config.h"
#ifdef _OPENCL
#include "shadowdetection/opencl/OpenCLImageParameters.h"
#endif

#define SPACES_COUNT 3
#define HSV_PARAMETERS 5
#define HLS_PARAMETERS 5
#define BGR_PARAMETERS 2
#define ROI_PARAMETERS 1;

namespace shadowdetection{
    namespace tools{
        namespace image{
            using namespace std;
            using namespace cv;
            using namespace core::opencv2;
            using namespace core::util;
            using namespace core::util::raii;
            using namespace shadowdetection::opencl;
            
            ImageShadowParameters::ImageShadowParameters(){
                regionsAvgsSecondChannel = 0;
                numOfSegments = 1;
            }
            
            ImageShadowParameters::~ImageShadowParameters(){
                reset();
            }
            
            float* ImageShadowParameters::merge(float label, const float** arrs, int arrsLen, int* arrSize, int& retSize) {
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
            
            float* ImageShadowParameters::merge(float** arrs, int arrsLen, int* arrSize, int& retSize){
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
            
            Matrix<float>* ImageShadowParameters::getImageParameters(const Mat& originalImage, const Mat& maskImage, 
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
                int maskChan = maskImage.channels();
                if (maskChan > 1){
                    SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "ImageParameters::getImageParameters, MaskImage");
                    throw exc;
                }
                int height = originalImage.size().height;
                int width = originalImage.size().width;
                int noLabelDataRowDimension;
                int pixelCount;
                
                Mat* hsv = OpenCV2Tools::convertToHSV(&originalImage);
                if (hsv == 0){
                    return 0;
                }
                ImageNewRaii hsvRaii(hsv);
                
                Mat* hls = OpenCV2Tools::convertToHLS(&originalImage);
                if (hls == 0){
                    return 0;
                }
                ImageNewRaii hlsRaii(hls);
                
                const Matrix<float>* noLabel = getImageParameters(  originalImage, *hsv, *hls,
                                                                    noLabelDataRowDimension, pixelCount);
                if (noLabel != 0){
                    PointerRaii< const Matrix<float> > noLabelRaii(noLabel);
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            float label = OpenCV2Tools::getChannelValue(maskImage, j, i, 0); //getLabel(maskImage.data[i * maskStep + j * maskChan]);
                            label /= 255.f;
                            
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
            
            Matrix<float>* ImageShadowParameters::getImageParameters( const Mat& originalImage, 
                                                                const Mat& hsvImage,
                                                                const Mat& hlsImage,
                                                                int& rowDimension,
                                                                int& pixelNum) throw (SDException&){
                if (originalImage.data == 0 || hsvImage.data == 0 || hlsImage.data == 0)
                    return 0;
                int height = originalImage.size().height;
                int width = originalImage.size().width;                
#ifdef _OPENCL
                pixelNum = width * height;
                uint parameterCount = HSV_PARAMETERS + HLS_PARAMETERS + BGR_PARAMETERS;
                Matrix<float>* ret = OpenCLImageParameters::getInstancePtr()->getImageParameters(&originalImage, 
                                                                            &hsvImage, &hlsImage, parameterCount);
                OpenCLImageParameters::getInstancePtr()->cleanWorkPart();
                rowDimension = parameterCount;
                return ret;
#else
                Matrix<float>* ret = 0;
                PointerRaii< Matrix<float> > retRaii;                                                                
                
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        KeyVal<uint> location((uint)j, (uint)i);
                        uchar hHSV = OpenCV2Tools::getChannelValue(hsvImage, location, 0);
                        uchar sHSV = OpenCV2Tools::getChannelValue(hsvImage, location, 1);
                        uchar vHSV = OpenCV2Tools::getChannelValue(hsvImage, location, 2);

                        uchar hHLS = OpenCV2Tools::getChannelValue(hlsImage, location, 0);
                        uchar lHLS = OpenCV2Tools::getChannelValue(hlsImage, location, 1);
                        uchar sHLS = OpenCV2Tools::getChannelValue(hlsImage, location, 2);

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
                        uchar B = OpenCV2Tools::getChannelValue(originalImage, location, 0);
                        uchar G = OpenCV2Tools::getChannelValue(originalImage, location, 1);
                        uchar R = OpenCV2Tools::getChannelValue(originalImage, location, 2);
                        procs[2] = processBGR(B, G, R, size[2]);
                        if (procs[2] == 0){
                            return 0;
                        }
                        VectorRaii vraiiProc2(procs[2]);                                                
                        
                        int mergedSize = 0;
                        float* merged = ImageShadowParameters::merge(procs, SPACES_COUNT, size, mergedSize);
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
#endif
            }
            
            float* ImageShadowParameters::processHSV(uchar H, uchar S, uchar V, int& size) {
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

            float* ImageShadowParameters::processHLS(uchar H, uchar L, uchar S, int& size) {
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
            
            float* ImageShadowParameters::processBGR(uchar B, uchar G, uchar R, int& size){
                size = BGR_PARAMETERS;
                float* retArr = MemMenager::allocate<float>(size);
                if (retArr != 0){
                    retArr[0] = (float)B / 255.f;
                    retArr[0] = clamp<float>(retArr[0], 0.f, 1.f);
                    retArr[1] = (float)(G + R) / (255.f + 255.f);
                    retArr[1] = clamp<float>(retArr[1], 0.f, 1.f);
                    //normalized bgr values
//                    float bgrSum = (float)((int)B + (int)G + (int)R + 1);
//                    retArr[2] = (float)B / bgrSum;
//                    retArr[2] = clamp<float>(retArr[2], 0.f, 1.f);
//                    retArr[3] = (float)G / bgrSum;
//                    retArr[3] = clamp<float>(retArr[3], 0.f, 1.f);
//                    retArr[4] = (float)R / bgrSum;
//                    retArr[4] = clamp<float>(retArr[4], 0.f, 1.f);
                }
                return retArr;
            }
            
            Matrix<float>* ImageShadowParameters::getAvgChannelValForRegions(const Mat* originalImage, uchar channelIndex){                                                
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
            
            float* ImageShadowParameters::processROI( KeyVal<uint> location, const Mat* originalImage, 
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
                
                float value = (float)OpenCV2Tools::getChannelValue(*originalImage, location, channelIndex);
                
                int yAvgIndex = location.getVal() / segmentHeight;
                int xAvgIndex = location.getKey() / segmentWidth;                
                    
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
            
            void ImageShadowParameters::reset(){
                if (regionsAvgsSecondChannel)
                    delete regionsAvgsSecondChannel;
                regionsAvgsSecondChannel = 0;
                numOfSegments = 1;
            }
            
        }
    }
}