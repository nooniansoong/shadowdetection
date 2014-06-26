#ifndef __IMAGE_PARAMETERS_H__
#define __IMAGE_PARAMETERS_H__

#include "opencv2/core/core.hpp"
#include "typedefs.h"

#define SPACES_COUNT 2
#define HSV_PARAMETERS 6
#define HLS_PARAMETERS 6


namespace shadowdetection{
    namespace util{
        namespace image{
            
            class ImageParameters{
            private:
            protected:
            public:
                static float* merge(float** arrs, int arrsLen, int* arrSize, int& retSize);
                static float* merge(float label, float** arrs, int arrsLen, int* arrSize, int& retSize);
                static float** getImageParameters(  cv::Mat& originalImage, int& rowDimension,
                                                    int& pixelNum) throw (SDException&);
                static float** getImageParameters(  cv::Mat& originalImage, cv::Mat& maskImage, 
                                                    int& rowDimension,
                                                    int& pixelNum) throw (SDException&);
                static float* processHSV(uchar H, uchar S, uchar V, int& size);
                static float* processHLS(uchar H, uchar L, uchar S, int& size);
            };
            
        }
    }
}

#endif