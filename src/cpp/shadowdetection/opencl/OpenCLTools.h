/* 
 * File:   openCLtols.h
 * Author: marko
 *
 * Created on May 28, 2014, 10:10 PM
 */

#ifndef __OPENCLTOOLS_H__
#define	__OPENCLTOOLS_H__

#ifdef _OPENCL

#include "core/opencl/OpenClToolsBase.h"
#include "core/util/Singleton.h"

struct svm_node;

namespace cv{
    class Mat;
}

namespace core{
    namespace util{
        template<typename T> class Matrix;
    }
}

namespace shadowdetection {
    namespace opencl {        

        class OpenclTools : public core::opencl::OpenClBase, public core::util::Singleton<OpenclTools>{
            friend class core::util::Singleton<OpenclTools>;
        private:
            cl_mem inputImage;
            cl_mem hsi1Converted;
            cl_mem hsi2Converted;
            cl_mem tsaiOutput;
            unsigned char* ratios1;
            unsigned char* ratios2;
                                    
            /**
             * create memory buffers for each kernel function
             * @param image
             * @param height
             * @param width
             * @param channels
             */
            void createBuffers(unsigned char* image, u_int32_t height, u_int32_t width, unsigned char channels);            
            /**
             * init global openCL variables
             */
            virtual void initVars();
            /**
             * init openCL variables necessary for one image processing
             */
            virtual void initWorkVars();
            /**
             * set arguments for kernel function1
             * @param height
             * @param width
             * @param channels
             */
            void setKernelArgs1(u_int32_t height, u_int32_t width, unsigned char channels, int lastKernelIndex);
            /**
             * set arguments for kernel function2
             * @param height
             * @param width
             * @param channels
             */
            void setKernelArgs2(u_int32_t height, u_int32_t width, unsigned char channels);
            /**
             * set arguments for kernel function3
             * @param height
             * @param width
             * @param channels
             */
            void setKernelArgs3(u_int32_t height, u_int32_t width, unsigned char channels);                        
        protected:
            OpenclTools();
            virtual std::string getClassName();
        public:            
            virtual ~OpenclTools();            
            /**
             * process image and returns binarized grayscale image with detected shadows (white color)
             * @param image
             * @param width
             * @param height
             * @param channels
             * @return 
             */
            cv::Mat* processRGBImage(unsigned char* image, u_int32_t width, u_int32_t height, unsigned char channels) throw (SDException&);            
            /**
             * clean up global variables
             */
            virtual void cleanUp();
            /**
             * clean up variables used for single image processing
             */
            virtual void cleanWorkPart();            
            //Image part
            uint32_t* convertHSI1(  uchar* image, u_int32_t width, u_int32_t height, 
                                    uchar channels) throw(SDException&);
            uint32_t* convertHSI2(  uchar* image, u_int32_t width, u_int32_t height, 
                                    uchar channels) throw(SDException&);                                
        };

    }
}

#endif

#endif	/* OPENCLTOLS_H */

