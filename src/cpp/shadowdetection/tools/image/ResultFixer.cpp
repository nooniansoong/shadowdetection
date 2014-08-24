#include "ResultFixer.h"
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
        
            ResultFixer::ResultFixer(){
                init();
            }
            
            ResultFixer::~ResultFixer(){
                
            }
            
            void ResultFixer::init() throw(SDException&){
                Config* conf = Config::getInstancePtr();                                
                string lThreshStr = conf->getPropertyValue("process.Thresholds.shadow.lValue");
                lThresh = (uchar)atoi(lThreshStr.c_str());
                string useThreshStr = conf->getPropertyValue("process.Thresholds.shadow.useThresh");
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
                
                SkyDetection skyDetection(originalImage);
                skyDetection.process();
                
                for (int i = 0; i < image.rows; i++){
                    for (int j = 0; j < image.cols; j++){
                        Pair<uint> location((uint)j, (uint)i);
                        uchar shadowValue = OpenCV2Tools::getChannelValue(image, location, 0);
                        if (shadowValue > 0){                            
                            uchar lValue = OpenCV2Tools::getChannelValue(hlsImage, location, 1);
                            if (lValue >= lThresh){                                
                                OpenCV2Tools::setChannelValue(image, location, 0, 0);
                            }
                            //sky detection
                            else if (skyDetection.isSky(location) == true){
                                OpenCV2Tools::setChannelValue(image, location, 0, 0);
                            }
                        }
                    }
                }                                
            }
            
        }
    }
}
