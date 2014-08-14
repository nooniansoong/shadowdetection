#include "SkyDetection.h"
#include "core/util/Config.h"
#include <string>
#include "core/util/raii/RAIIS.h"

namespace skydetection{
    
    using namespace cv;
    using namespace core::util;
    using namespace std;
    using namespace core::opencv2;
    using namespace core::util::raii;
    
    void SkyDetection::initBaseVariables(){
        originalImage = 0;
        detectedImage = 0;
        
        Config* conf = Config::getInstancePtr();
        string rThreshStr = conf->getPropertyValue("process.Thresholds.sky.rValue");
        rThresh = (uchar)atoi(rThreshStr.c_str());
        string bThreshStr = conf->getPropertyValue("process.Thresholds.sky.bValue");
        bThresh = (uchar)atoi(bThreshStr.c_str());
        string lThreshStr = conf->getPropertyValue("process.Thresholds.sky.lValue");
        lThresh = (uchar)atoi(lThreshStr.c_str());
    }
    
    SkyDetection::SkyDetection(){
        initBaseVariables();
    }
    
    SkyDetection::SkyDetection(const Mat& originalImage){
        initBaseVariables();
        this->originalImage = new Mat(originalImage);
    }
    
    SkyDetection::~SkyDetection(){
        if (originalImage != 0)
            delete originalImage;
        if (detectedImage != 0)
            delete detectedImage;
    }
    
    void SkyDetection::process() throw (SDException&){
        if (originalImage == 0 || originalImage->data == 0){
            SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "SkyDetection::isSky");
            throw exc;
        }
        detectedImage = OpenCV2Tools::get8bitImage(originalImage->rows, originalImage->cols);
        if (detectedImage == 0 || detectedImage->data == 0){
            SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "SkyDetection::isSky");
            throw exc;
        }
        Mat* hlsImage = OpenCV2Tools::convertToHLS(originalImage);
        if (hlsImage == 0 || hlsImage->data == 0){
            SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "SkyDetection::isSky");
            throw exc;
        }
        ImageNewRaii imageRaii(hlsImage);
        for (int i = 0; i < originalImage->rows; i++){
            for (int j = 0; j < originalImage->cols; j++){
                KeyVal<uint> location((uint)j, (uint)i);
                uchar rValue = OpenCV2Tools::getChannelValue(*originalImage, location, 2);
                uchar gValue = OpenCV2Tools::getChannelValue(*originalImage, location, 1);
                uchar bValue = OpenCV2Tools::getChannelValue(*originalImage, location, 0);
                uchar lValue = OpenCV2Tools::getChannelValue(*hlsImage, location, 1);
                if (rValue <= rThresh && (gValue >= bValue / 4U) && 
                    gValue <= bValue && bValue >= bThresh && 
                    lValue >= lThresh){
                        OpenCV2Tools::setChannelValue(*detectedImage, location, 0, 255U);
                }
                else{
                    OpenCV2Tools::setChannelValue(*detectedImage, location, 0, 0U);
                }
            }
        }
    }
    
    bool SkyDetection::isSky(KeyVal<uint> location) throw (SDException&){
        if (detectedImage == 0 || detectedImage->data == 0){
            SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "SkyDetection::isSky");
            throw exc;
        }
        uchar skyValue = OpenCV2Tools::getChannelValue(*detectedImage, location, 0);
        if (skyValue != 0U)
            return true;
        return false;
    }
    
    Mat* SkyDetection::getDetected(){
        return detectedImage;
    }
    
}
