#ifndef __IMAGE_PARAMETERS_H__
#define __IMAGE_PARAMETERS_H__

#include "opencv2/core/core.hpp"
#include "typedefs.h"
#include "shadowdetection/util/Matrix.h"

namespace shadowdetection{
    namespace util{
        namespace image{
            
            class ImageParameters{
            private:
                shadowdetection::util::Matrix<float>* regionsAvgsSecondChannel;
                int numOfSegments;
                float segmentWidth;
                float segmentHeight;
                
                static float* processHSV(   uchar H, uchar S, uchar V, int& size);
                static float* processHLS(   uchar H, uchar L, uchar S, int& size);
                static float* processBGR(   uchar B, uchar G, uchar R, int& size);
                float* processROI(  KeyVal<uint> location, const cv::Mat* originalImage, 
                                    int& size, uchar channelIndex) throw (SDException&);
                shadowdetection::util::Matrix<float>* getAvgChannelValForRegions(const cv::Mat* originalImage,
                                                                                uchar channelIndex);
            protected:
            public:
                ImageParameters();
                virtual ~ImageParameters();
                
                static float* merge(float** arrs, int arrsLen, int* arrSize, int& retSize);
                static float* merge(float label, const float** arrs, int arrsLen, int* arrSize, int& retSize);
                shadowdetection::util::Matrix<float>* getImageParameters(   const cv::Mat& originalImage,
                                                                            const cv::Mat& hsvImage,
                                                                            const cv::Mat& hlsImage,
                                                                            int& rowDimension, int& pixelNum) throw (SDException&);
                shadowdetection::util::Matrix<float>* getImageParameters(  const cv::Mat& originalImage, const cv::Mat& maskImage,                                                     
                                                    int& rowDimension, int& pixelNum) throw (SDException&);
                void reset();
            };
            
        }
    }
}

#endif