/* 
 * File:   OpenCV2Tools.h
 * Author: marko
 *
 * Created on June 6, 2014, 1:06 PM
 */

#ifndef OPENCV2TOOLS_H
#define	OPENCV2TOOLS_H

#include "opencv2/ocl/ocl.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "typedefs.h"

namespace shadowdetection {
    namespace opencv2 {

        class OpenCV2Tools {
        private:
        protected:
        public:
            static uint* convertImagetoHSI  (const cv::Mat* image, int& height, int& width, int& channels,
                                             void (*convertFunc)(unsigned char, unsigned char, unsigned char, unsigned int&, unsigned char&, unsigned char&));
            /**
            * 
            * @param image
            * @param copy 
             * if copy == true expect BGR format not RGB
            * @return 
            */
            static unsigned char* convertImageToByteArray(const cv::Mat* image, bool copy = false);
            /**
             * creates image header for arr data. 
             * @param arr
             * arr should exists until working with image
             * @param width
             * @param height
             * @param channels
             * @return 
             */
            static cv::Mat* convertByteArrayToImage(unsigned char* arr, int width, int height, int channels);
            /**
             * creates image header for single channel image
             * @param input
             * input should exists until working with image
             * @param height
             * @param width
             * @return 
             */
            static cv::Mat* get8bitImage(unsigned char* input, int height, int width);
            static cv::Mat* binarize(const cv::Mat* input);
            static cv::Mat* joinTwo(const cv::Mat* src1, const cv::Mat* src2);
#ifdef _OPENCL
            static void initOpenCL(int pid, int device) throw (SDException&);
            //OTZU still not supported
            //static cv::Mat* binarizeOcl(const cv::Mat& image);
            static cv::Mat* joinTwoOcl(const cv::Mat& src1, const cv::Mat& src2);
#endif
        };

    }
}
#endif	/* OPENCV2TOOLS_H */

