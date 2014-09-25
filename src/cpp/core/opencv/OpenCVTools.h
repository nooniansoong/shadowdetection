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
             * input: value of r channel pixel to be converted
             * @param g
             * input: value of g channel pixel to be converted
             * @param b
             * input: value of r channel pixel to be converted
             * @param h
             * output: h channel value of converted pixel
             * @param s
             * output: s channel value of converted pixel
             * @param i
             * output: i channel value of converted pixel
             */
            static void RGBtoHSI_1          (unsigned char r, unsigned char g, unsigned char b, unsigned int& h, unsigned char& s, unsigned char& i);
            /**
             * converts pixel to HSI second way
             * @param r
             * input: value of r channel pixel to be converted
             * @param g
             * input: value of g channel pixel to be converted
             * @param b
             * input: value of r channel pixel to be converted
             * @param h
             * output: h channel value of converted pixel
             * @param s
             * output: s channel value of converted pixel
             * @param i
             * output: i channel value of converted pixel             
             */
            static void RGBtoHSI_2          (unsigned char r, unsigned char g, unsigned char b, unsigned int& h, unsigned char& s, unsigned char& i);
            /**
             * converts BGR image to HSI
             * @param image
             * input IplImage struct pointer
             * @param height
             * output image height;
             * @param width
             * output image width
             * @param channels
             * output image number of channels
             * @param convertFunc
             * pointer to convert function
             * @return 
             * uint array where channels number of values represents one pixel, and each value represents channel value in H, S, I order
             */
            static unsigned int* convertImagetoHSI  (IplImage* image, int& height, int& width, int& channels,
                                             void (*convertFunc)(unsigned char, unsigned char, unsigned char, unsigned int&, unsigned char&, unsigned char&));
            /**
             * converts BGR image to RGB byte array
             * @param image
             * input IplImage struct pointer
             * @return 
             * byte array representing image and channels are in R, G , B order. return array is copy of data from image not a pointer to
             */
            static unsigned char* convertImageToByteArray(IplImage* image);
            /**
             * convert RGB byte array to BGR image
             * @param arr
             * uchar array where each pixel is represented with channels number of values, and values are representing channels in R, G, B order
             * @param width
             * width of input/output image
             * @param height
             * height of input/output image
             * @param channels
             * number of channels of input/output image
             * @return 
             * pointer to IplImage struct
             */
            static IplImage* convertByteArrayToImage(unsigned char* arr, int width, int height, int channels);
            /**
             * determing H / I proportions in image
             * @param inputHSI
             * input uint array representing image, where each pixel is represented with channels values, and channels are in H, S, I order
             * @param height
             * height of input image
             * @param width
             * width of input image
             * @param channels
             * number of channels of input image
             * @return 
             * proportions of H and I channels values for each pixel
             */
            static uchar* simpleTsai(unsigned int* inputHSI, int height, int width, int channels);
            /**
             * creates singel channel image from single channel byte array
             * @param input
             * single channel image represented by uchar array. Each pixel is represented by each array value
             * @param height
             * input image height
             * @param width
             * input image width
             * @return 
             * pointer to single channel IplImage struct
             */
            static IplImage* get8bitImage(unsigned char* input, int height, int width);
            /**
             * binarizes image using otsu's method
             * @param input
             * pointer to single channel IplImage struct
             * @return 
             * binarized image by otsu's method, as pointer to single channel IplImage struct
             */
            static IplImage* binarize(IplImage* input);
            /**
             * merges two single channel images using or operator
             * @param src1
             * first image, pointer to IplImage struct
             * @param src2
             * second image, pointer to IplImage struct
             * @return 
             * merge of dirst and second image using OR operator, represented as single channel pointer to IplImage
             */
            static IplImage* joinTwo(IplImage* src1, IplImage* src2);                                    
        };
    }
}

#endif	/* OPENCVTOOLS_H */

