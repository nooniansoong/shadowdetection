#ifndef __SKY_DETECTION_H__ 
#define __SKY_DETECTION_H__

#include "typedefs.h"
#include "core/opencv/OpenCV2Tools.h"

namespace skydetection{
    
    /**
     conatins a copy of originalImage
     */
    class SkyDetection{
    private:
        uchar rThresh;        
        uchar bThresh;
        uchar lThresh;
        
        cv::Mat* originalImage;
        cv::Mat* detectedImage;
        
        void initBaseVariables();
    protected:
    public:
        SkyDetection();
        SkyDetection(const cv::Mat& originalImage);
        virtual ~SkyDetection();
        
        void process() throw (SDException&);
        bool isSky(KeyVal<uint> location) throw (SDException&);
        /**
         * keep insatnce until using detected image
         * @return 
         */
        cv::Mat* getDetected();
    };
    
}

#endif
