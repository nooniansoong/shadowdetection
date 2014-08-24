#ifndef __SKY_DETECTION_H__ 
#define __SKY_DETECTION_H__

#include "typedefs.h"
#include "core/opencv/OpenCV2Tools.h"
#include "hash_set"

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
        void processSegments();
        Triple<float> getMeanBGRValuesOfSegment(__gnu_cxx::hash_set< Pair<uint> >* segment);
        void reduceInSegment(__gnu_cxx::hash_set< Pair<uint> >* segment, const Triple<float>& thresHold);
    protected:
    public:
        SkyDetection();
        SkyDetection(const cv::Mat& originalImage);
        virtual ~SkyDetection();
        
        void process() throw (SDException&);
        bool isSky(Pair<uint> location) throw (SDException&);
        /**
         * keep insatnce until using detected image
         * @return 
         */
        cv::Mat* getDetected();
    };
    
}

#endif
