/* 
 * File:   OpenCVTools.h
 * Author: marko
 *
 * Created on May 30, 2014, 8:00 PM
 */

#ifndef OPENCVTOOLS_H
#define	OPENCVTOOLS_H

#include <cv.h>
#include <highgui.h>
#include <math.h>

#include "opencv2/ocl/ocl.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace shadowdetection{
    namespace opencv{
        class OpenCvTools{
        private:
        protected:
        public:
            static void RGBtoHSI_1          (unsigned char r, unsigned char g, unsigned char b, unsigned int& h, unsigned char& s, unsigned char& i);
            static void RGBtoHSI_2          (unsigned char r, unsigned char g, unsigned char b, unsigned int& h, unsigned char& s, unsigned char& i);
            static uint* convertImagetoHSI  (IplImage* image, int& height, int& width, int& channels,
                                             void (*convertFunc)(unsigned char, unsigned char, unsigned char, unsigned int&, unsigned char&, unsigned char&));
            static unsigned char* convertImageToByteArray(IplImage* image);
            static IplImage* convertByteArrayToImage(unsigned char* arr, int width, int height, int channels);
            static uchar* simpleTsai(unsigned int* inputHSI, int height, int width, int channels);
            static IplImage* get8bitImage(unsigned char* input, int height, int width);
            static IplImage* binarize(IplImage* input);
            static IplImage* joinTwo(IplImage* src1, IplImage* src2);
                        
            //static IplImage* convertMatToIplImage(const cv::Mat& matrix);
#ifdef _OPENCL
            static void initOpenCL(int pid, int device) throw (int);
            //OTZU still not supported
            //static cv::Mat* binarizeOcl(const cv::Mat& image);
            static cv::Mat* joinTwoOcl(const cv::Mat& src1, const cv::Mat& src2);
#endif
        };
    }
}

#endif	/* OPENCVTOOLS_H */

