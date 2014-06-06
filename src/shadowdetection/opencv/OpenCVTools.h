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
        };
    }
}

#endif	/* OPENCVTOOLS_H */

