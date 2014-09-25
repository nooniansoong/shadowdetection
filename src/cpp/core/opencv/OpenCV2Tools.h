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
#include <list>
#include <unordered_set>
#include "typedefs.h"

namespace core{
    namespace opencv2{

        class OpenCV2Tools {
        private:
        protected:
        public:
            /**
             * converts BGR image to HSI
             * @param image
             * input Mat object pointer
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
            static unsigned int* convertImagetoHSI  (const cv::Mat* image, int& height, int& width, int& channels,
                                             void (*convertFunc)(unsigned char, unsigned char, unsigned char, unsigned int&, unsigned char&, unsigned char&));
            /**
             * converts BGR image to RGB byte array
             * @param image
             * input Mat object pointer
             * @return 
             * byte array representing image and channels are in R, G , B order. return array is copy of data from image not a pointer to
             */
            static unsigned char* convertImageToByteArray(const cv::Mat* image, bool copy = false);
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
             * pointer to Mat object
             */
            static cv::Mat* convertByteArrayToImage(unsigned char* arr, int width, int height, int channels);
            /**
             * creates single channel image from single channel byte array
             * @param input
             * single channel image represented by uchar array. Each pixel is represented by each array value
             * @param height
             * input image height
             * @param width
             * input image width
             * @return 
             * pointer to single channel Mat object
             */
            static cv::Mat* get8bitImage(unsigned char* input, int height, int width);            
            /**
             * creates empty single channel 8bit image
             * @param input
             * single channel image represented by uchar array. Each pixel is represented by each array value
             * @param height
             * input image height
             * @param width
             * input image width
             * @return 
             * pointer to single channel Mat object
             */
            static cv::Mat* get8bitImage(int height, int width);
            /**
             * creates empty single channel 24bit image
             * @param input
             * three channel image represented by uchar array. Each pixel is represented by three array values
             * @param height
             * input image height
             * @param width
             * input image width
             * @return 
             * pointer to single channel Mat object
             */
            static cv::Mat* get24bitImage(int height, int width);
            /**
             * binarizes image using otsu's method
             * @param input
             * pointer to single channel Mat object
             * @return 
             * binarized image by otsu's method, as pointer to single channel Mat object
             */
            static cv::Mat* binarize(const cv::Mat* input);
            /**
             * merges two single channel images using or operator
             * @param src1
             * first image, pointer to Mat object
             * @param src2
             * second image, pointer to Mat object
             * @return 
             * merge of first and second image using OR operator, represented as single channel pointer to Mat object
             */
            static cv::Mat* joinTwo(const cv::Mat* src1, const cv::Mat* src2);
            /**
             * 
             * @param src
             * input image as pointer to Mat object, BGR channel order
             * @return
             * image as pointer to Mat object HSV channel order 
             */
            static cv::Mat* convertToHSV(const cv::Mat* src) throw (SDException&);
            /**
             * 
             * @param src
             * input image as pointer to Mat object, BGR channel order
             * @return
             * pointer to Mat object representing image in HLS format 
             */
            static cv::Mat* convertToHLS(const cv::Mat* src) throw (SDException&);
            /**
             * return region of interest for specified image. Region is specified by location and dimensions.
             * Returned image points to original image data.
             * Method also doing image size check so result area can be smaller of specified
             * @param src
             * input image as pointer to Mat object
             * @param roiWidth
             * width of region
             * @param roiHeight
             * height of region
             * @param location
             * region top left location
             * @return 
             * Mat object representing region
             */
            static cv::Mat* getImageROI(const cv::Mat* src, uint roiWidth, uint roiHeight,
                                        const Pair<uint>& location) throw (SDException&);
            /**
             * 
             * @param src
             * input image as pointer to Mat object
             * @param channelIndex
             * index of channel for which doing operation, must be less of image num of channels
             * @return
             * average channel value on whole image 
             */
            static float getAvgChannelValue(const cv::Mat& src, 
                                            uchar channelIndex) throw (SDException&);
            /**
             * 
             * @param image
             * input image as pointer to Mat object
             * @param x
             * x coordinate of location of pixel in image
             * @param y
             * y coordinate of location of pixel in image
             * @param channelIndex
             * index of channel for which doing operation, must be less of image num of channels 
             * @return 
             * return channelIndex channel value of pixel specified by location 
             */
            static uchar getChannelValue(   const cv::Mat& image, uint x, uint y, 
                                            uchar channelIndex) throw (SDException&);
            /**
             * 
             * @param image
             * input image as pointer to Mat object
             * @param location
             * location of pixel in image
             * @param channelIndex
             * index of channel for which doing operation, must be less of image num of channels
             * @return
             * return channelIndex channel value of pixel specified by location 
             */
            static uchar getChannelValue(   const cv::Mat& image, Pair<uint> location, 
                                            uchar channelIndex) throw (SDException&);
            /**
             * 
             * @param image
             * input image as pointer to Mat object
             * @param location
             * Location of pixel for which doing operation
             * @param channelIndex
             * channel index, must be less of image num of channels
             * @param newValue
             * new value of channel of pixel specified by location
             */
            static void setChannelValue(cv::Mat& image, Pair<uint> location, 
                                        uchar channelIndex, uchar newValue) throw (SDException&);
            /**
             * Segmentize image by value of first channel.
             * @param image
             * input image as pointer to Mat object
             * @param color
             * value of first channel by which image will be segmentised
             * @return 
             * List of unconnected regions. Each region is set of connected pixels, and each pixel's first channel is equals to color value
             */
            static std::list< std::unordered_set< Pair<uint> >* >* getRegionsOfColor(const cv::Mat& image, 
                                                                                        const uint& color) throw (SDException&);
            /**
             * 
             * @param segments
             * segments which will be freed up from memory
             * Segments are returned from getRegionsOfColor() method
             */
            static void destroySegments(std::list< std::unordered_set< Pair<uint> >* >* segments) throw (SDException&);
#ifdef _OPENCL
            /**
             * init global variables needed for openCV openCL processing
             * @param pid
             * openCL platform ID
             * @param device
             * OpenCl platform device ID
             */
            static void initOpenCL(uint pid, uint device) throw (SDException&);
            //OTZU still not supported
            //static cv::Mat* binarizeOcl(const cv::Mat& image);
            /**
             * merges two single channel images using or operator, using OpenCL
             * @param src1
             * first image, pointer to Mat object
             * @param src2
             * second image, pointer to Mat object
             * @return 
             * merge of first and second image using OR operator, represented as single channel pointer to Mat object
             */
            static cv::Mat* joinTwoOcl(const cv::Mat& src1, const cv::Mat& src2);
#endif
        };

    }
}
#endif	/* OPENCV2TOOLS_H */

