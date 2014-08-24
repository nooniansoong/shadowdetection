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

namespace core{
    namespace opencv{
        class OpenCvTools{
        private:
        protected:
        public:
            /**
             * converts pixel to HSI first way
             * @param r
             * @param g
             * @param b
             * @param h
             * @param s
             * @param i
             */
            static void RGBtoHSI_1          (unsigned char r, unsigned char g, unsigned char b, unsigned int& h, unsigned char& s, unsigned char& i);
            /**
             * converts pixel to HSI second way
             * @param r
             * @param g
             * @param b
             * @param h
             * @param s
             * @param i
             */
            static void RGBtoHSI_2          (unsigned char r, unsigned char g, unsigned char b, unsigned int& h, unsigned char& s, unsigned char& i);
            /**
             * converts BGR image to HSI
             * @param image
             * @param height
             * @param width
             * @param channels
             * @param convertFunc
             * @return 
             */
            static unsigned int* convertImagetoHSI  (IplImage* image, int& height, int& width, int& channels,
                                             void (*convertFunc)(unsigned char, unsigned char, unsigned char, unsigned int&, unsigned char&, unsigned char&));
            /**
             * converts BGR image to RGB byte array
             * @param image
             * @return 
             */
            static unsigned char* convertImageToByteArray(IplImage* image);
            /**
             * convert RGB byte array to BGR image
             * @param arr
             * @param width
             * @param height
             * @param channels
             * @return 
             */
            static IplImage* convertByteArrayToImage(unsigned char* arr, int width, int height, int channels);
            /**
             * determing H / I proportions in image
             * @param inputHSI
             * @param height
             * @param width
             * @param channels
             * @return 
             */
            static uchar* simpleTsai(unsigned int* inputHSI, int height, int width, int channels);
            /**
             * creates singel channel image from single channel byte array
             * @param input
             * @param height
             * @param width
             * @return 
             */
            static IplImage* get8bitImage(unsigned char* input, int height, int width);
            /**
             * binarizes image using otzu
             * @param input
             * @return 
             */
            static IplImage* binarize(IplImage* input);
            /**
             * merges two single channel images using or operator
             * @param src1
             * @param src2
             * @return 
             */
            static IplImage* joinTwo(IplImage* src1, IplImage* src2);                                    
        };
    }
}

#endif	/* OPENCVTOOLS_H */

