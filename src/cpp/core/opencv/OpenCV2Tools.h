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
             * @param height
             * @param width
             * @param channels
             * @param convertFunc
             * @return 
             */
            static unsigned int* convertImagetoHSI  (const cv::Mat* image, int& height, int& width, int& channels,
                                             void (*convertFunc)(unsigned char, unsigned char, unsigned char, unsigned int&, unsigned char&, unsigned char&));
            /**
            * convert image to RGB byte array
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
            /**
             * creates empty one channel 8bit image             
             * @return 
             */
            static cv::Mat* get8bitImage(int height, int width);
            /**
             * creates empty three channel 8bit image             
             * @return 
             */
            static cv::Mat* get24bitImage(int height, int width);
            /**
             * binarize image using otzu
             * @param input
             * @return 
             */
            static cv::Mat* binarize(const cv::Mat* input);
            /**
             * joins two single channel images using or operator
             * @param src1
             * @param src2
             * @return 
             */
            static cv::Mat* joinTwo(const cv::Mat* src1, const cv::Mat* src2);
            static cv::Mat* convertToHSV(const cv::Mat* src) throw (SDException&);
            static cv::Mat* convertToHLS(const cv::Mat* src) throw (SDException&);
            static cv::Mat* getImageROI(const cv::Mat* src, uint roiWidth, uint roiHeight,
                                        const Pair<uint>& location) throw (SDException&);
            static float getAvgChannelValue(const cv::Mat* src, 
                                            uchar channelIndex) throw (SDException&);
            static uchar getChannelValue(   const cv::Mat& image, uint x, uint y, 
                                            uchar channelIndex) throw (SDException&);
            static uchar getChannelValue(   const cv::Mat& image, Pair<uint> location, 
                                            uchar channelIndex) throw (SDException&);
            static void setChannelValue(cv::Mat& image, Pair<uint> location, 
                                        uchar channelIndex, uchar newValue) throw (SDException&);
            static std::list< std::unordered_set< Pair<uint> >* >* getRegionsOfColor(const cv::Mat& image, 
                                                                                        const uint& color) throw (SDException&);
            static void destroySegments(std::list< std::unordered_set< Pair<uint> >* >* segments) throw (SDException&);
#ifdef _OPENCL
            /**
             * init global variables needed for openCV openCL processing
             * @param pid
             * @param device
             */
            static void initOpenCL(uint pid, uint device) throw (SDException&);
            //OTZU still not supported
            //static cv::Mat* binarizeOcl(const cv::Mat& image);
            /**
             * joins two single channel images using or operator using openCL
             * @param src1
             * @param src2
             * @return 
             */
            static cv::Mat* joinTwoOcl(const cv::Mat& src1, const cv::Mat& src2);
#endif
        };

    }
}
#endif	/* OPENCV2TOOLS_H */

