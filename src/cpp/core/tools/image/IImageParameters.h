#ifndef __I_IMAGE_PARAMETERS_H__
#define __I_IMAGE_PARAMETERS_H__

#include "core/util/Matrix.h"
#include "core/opencv/OpenCV2Tools.h"

namespace core{
    namespace tools{
        namespace image{
            
            class IImageParameteres{
            private:
            protected:
            public:
                virtual core::util::Matrix<float>* getImageParameters(  const cv::Mat& originalImage,
                                                                        const cv::Mat& hsvImage,
                                                                        const cv::Mat& hlsImage,
                                                                        int& rowDimension, int& pixelNum) throw (SDException&) = 0;
                virtual core::util::Matrix<float>* getImageParameters(  const cv::Mat& originalImage, 
                                                                        const cv::Mat& maskImage,                                                     
                                                                        int& rowDimension, int& pixelNum) throw (SDException&) = 0;
                virtual void reset() = 0;
            };
            
        }
    }
}

#endif
