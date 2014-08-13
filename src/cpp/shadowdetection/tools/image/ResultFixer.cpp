#include "ResultFixer.h"
#include "core/opencv/OpenCV2Tools.h"
#include "core/util/Config.h"

namespace shadowdetection{
    namespace tools{
        namespace image{
            
            using namespace cv;
            using namespace core::opencv2;
            using namespace core::util;
        
            ResultFixer::ResultFixer(){
                init();
            }
            
            ResultFixer::~ResultFixer(){
                
            }
            
            void ResultFixer::init() throw(SDException&){
                Config* conf = Config::getInstancePtr();
                string rThreshStr = conf->getPropertyValue("process.Thresholds.Values.rValue");
                rThresh = (uchar)atoi(rThreshStr.c_str());
//                string gThreshStr = conf->getPropertyValue("process.Thresholds.Values.gValue");
//                gThresh = (uchar)atoi(gThreshStr.c_str());
                string bThreshStr = conf->getPropertyValue("process.Thresholds.Values.bValue");
                bThresh = (uchar)atoi(bThreshStr.c_str());                
                string lThreshStr = conf->getPropertyValue("process.Thresholds.Values.lValue");
                lThresh = (uchar)atoi(lThreshStr.c_str());
                string useThreshStr = conf->getPropertyValue("process.Thresholds.UseThresh");
                useThresh = true;
                if (useThreshStr == "false")
                    useThresh = false;                
            }
            
            void ResultFixer::applyThreshholds( Mat& image, const Mat& originalImage, 
                                                const Mat& hlsImage) throw(SDException&){
                if (image.channels() > 1){
                    SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "ResultFixer::applyThreshholds chn");
                    throw exc;
                }
                if (originalImage.channels() < 3 || hlsImage.channels() < 3){
                    SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "ResultFixer::applyThreshholds or chn");
                    throw exc;
                }
                
                if (image.size != originalImage.size || image.size != hlsImage.size){
                    SDException exc(SHADOW_INVALID_IMAGE_FORMAT, "ResultFixer::applyThreshholds sizes");
                    throw exc;
                }
                
                for (int i = 0; i < image.rows; i++){
                    for (int j = 0; j < image.cols; j++){
                        KeyVal<uint> location((uint)j, (uint)i);
                        uchar shadowValue = OpenCV2Tools::getChannelValue(image, location, 0);
                        if (shadowValue > 0){
                            uchar rValue = OpenCV2Tools::getChannelValue(originalImage, location, 2);
                            uchar gValue = OpenCV2Tools::getChannelValue(originalImage, location, 1);
                            uchar bValue = OpenCV2Tools::getChannelValue(originalImage, location, 0);
                            uchar lValue = OpenCV2Tools::getChannelValue(hlsImage, location, 1);
                            if (lValue >= lThresh){                                
                                OpenCV2Tools::setChannelValue(image, location, 0, 0);
                            }
                            //sky detection
                            else if (rValue <= rThresh && (gValue >= bValue / 4U) && 
                                    gValue <= bValue && bValue >= bThresh && 
                                    lValue >= lThresh / 3){
                                OpenCV2Tools::setChannelValue(image, location, 0, 0);
                            }
                        }
                    }
                }                                
            }
            
        }
    }
}
