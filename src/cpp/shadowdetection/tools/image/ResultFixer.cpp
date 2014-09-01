#include "ResultFixer.h"
#include <memory>
#include "core/opencv/OpenCV2Tools.h"
#include "core/util/Config.h"
#include "skydetection/SkyDetection.h"

namespace shadowdetection{
    namespace tools{
        namespace image{
            
            using namespace cv;
            using namespace core::opencv2;
            using namespace core::util;
            using namespace skydetection;
            using namespace std;
        
            ResultFixer::ResultFixer(){
                init();
            }
            
            ResultFixer::~ResultFixer(){
                
            }
            
            void ResultFixer::init() throw(SDException&){
                Config* conf = Config::getInstancePtr();                                
                string lThreshStr = conf->getPropertyValue("shadowDetection.Thresholds.lValue");
                lThresh = (uchar)atoi(lThreshStr.c_str());
                string useThreshStr = conf->getPropertyValue("shadowDetection.useThresholds");
                useThresh = true;
                if (useThreshStr == "false")
                    useThresh = false;
                useSky = true;
                string useSkyStr = conf->getPropertyValue("shadowDetection.useSkyDetection");
                if (useSkyStr == "false")
                    useSky = false;
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
                
                UNIQUE_PTR(SkyDetection) skyDetection;
                if (useSky){
                    skyDetection = UNIQUE_PTR(SkyDetection)(New SkyDetection(originalImage));
                    skyDetection->process();
                }
                
                if (useThresh || useSky){
                    for (int i = 0; i < image.rows; i++){
                        for (int j = 0; j < image.cols; j++){
                            Pair<uint> location((uint)j, (uint)i);
                            uchar shadowValue = OpenCV2Tools::getChannelValue(image, location, 0);
                            if (shadowValue > 0){                            
                                if (useThresh){
                                    uchar lValue = OpenCV2Tools::getChannelValue(hlsImage, location, 1);
                                    if (lValue >= lThresh){                                
                                        OpenCV2Tools::setChannelValue(image, location, 0, 0);
                                    }
                                }
                                //sky detection
                                if (useSky){
                                    if (skyDetection->isSky(location) == true){
                                        OpenCV2Tools::setChannelValue(image, location, 0, 0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
        }
    }
}
