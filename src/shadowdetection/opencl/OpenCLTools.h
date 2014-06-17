/* 
 * File:   openCLtols.h
 * Author: marko
 *
 * Created on May 28, 2014, 10:10 PM
 */

#ifndef OPENCLTOLS_H
#define	OPENCLTOLS_H

#ifdef _OPENCL

#ifndef _MAC
#include <CL/cl.h>
#else
#include <OpenCL/opencl.h>
#endif
#include "typedefs.h"
#include "shadowdetection/util/Singleton.h"

#define MAX_DEVICES 100
#define MAX_SRC_SIZE 5242800
#define KERNEL_COUNT 5
#define MAX_PLATFORMS 100
#define PROGRAM_COUNT 2

struct svm_node;

namespace cv{
    class Mat;
}

namespace shadowdetection {
    namespace opencl {        

        class OpenclTools : public shadowdetection::util::Singleton<OpenclTools>{
            friend class shadowdetection::util::Singleton<OpenclTools>;
        private:            
            cl_device_id device;
            cl_int err;
            cl_command_queue command_queue[PROGRAM_COUNT];
            cl_program program[PROGRAM_COUNT];
            cl_kernel kernel[KERNEL_COUNT];
            cl_context context[PROGRAM_COUNT];
            size_t workGroupSize[KERNEL_COUNT];
            cl_mem input;
            cl_mem output1;
            cl_mem output2;
            cl_mem output3;
            unsigned char* ratios1;
            unsigned char* ratios2;
            
            size_t shrRoundUp(size_t localSize, size_t allSize);
            /**
             * check for openCL error
             * @param err
             * @param err_code
             */
            void err_check(int err, std::string err_code, int programIndex) throw (SDException&);
            /**
             * create openCL kernels from program
             */
            void createKernels(int index);
            /**
             * calculate work group sizes for each kernel
             */
            void createWorkGroupSizes();
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
            void initVars();
            /**
             * init openCL variables necessary for one image processing
             */
            void initWorkVars();
            /**
             * set arguments for kernel function1
             * @param height
             * @param width
             * @param channels
             */
            void setKernelArgs1(u_int32_t height, u_int32_t width, unsigned char channels);
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
            
            /**
             * global function for load program
             * @param kernelFileName
             */
            void loadKernelFile(std::string& kernelFileName, int index);
            /**
             * load program from source
             * @param kernelFileName
             */
            void loadKernelFileFromSource(std::string& kernelFileName, int index);
            /**
             * load program from precompiled binary
             * @param kernelFileName
             * @return 
             */
            bool loadKernelFileFromBinary(std::string& kernelFileName, int index);
            /**
             * saves program binary loaded from source
             */
            const char* saveKernelBinary(std::string& kernelFileName, int index);
        protected:
            OpenclTools();
        public:            
            virtual ~OpenclTools();
            /**
             * init variables for OpenclTools class instances
             * @param platformID
             * @param deviceID
             * @param listOnly
             */
            void init(int platformID, int deviceID, bool listOnly) throw (SDException&);
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
             * process image and returns binarized grayscale image with detected shadows (white color)
             * using mostly openCV C API
             * @param image
             * @param width
             * @param height
             * @param channels
             * @return 
             */
            cv::Mat* processRGBImageOld(unsigned char* image, u_int32_t width, u_int32_t height, unsigned char channels) throw (SDException&);
            /**
             * clean up global variables
             */
            void cleanUp();
            /**
             * clean up variables used for single image processing
             */
            void cleanWorkPart();
            //libsvm section
        private:
            struct cl_svm_node{
                cl_int index;
                cl_double value;
            };
            
            cl_mem clData;
            int clDataLen;
            cl_mem clY;
            cl_mem clX;
            bool newTask;
            
            svm_node *xForCl;
            void createBuffersSVM(  float* data, int dataLen,
                                    char* y, int yLen,
                                    const svm_node* x, int xX, int xY, bool diffX,
                                    int start, int steps, bool& clDataChanged);
            void setKernelArgsSVC( cl_int start, cl_int len, cl_int i, 
                                    cl_int kernel_type, cl_int xW, cl_int dataLen,
                                    cl_double gamma, cl_double coef0, cl_int degree,
                                    bool clDataChanged);
            void setKernelArgsSVR(  cl_int start, cl_int len, cl_int i, 
                                    cl_int kernel_type, cl_int xW, cl_int dataLen,
                                    cl_double gamma, cl_double coef0, cl_int degree,
                                    bool clDataChanged);
            bool xDif (const svm_node** x, int xLen, int xW);
        protected:
        public:
            void get_Q( float* data, int dataLen, int start, int len, int i, int kernel_type, 
                        char* y, int yLen, const svm_node** x, int xX, int xY, LIBSVM_CLASS_TYPE classType,
                        double gamma, double coef0, int degree, double *x_square) throw (SDException);            
        };

    }
}

#endif

#endif	/* OPENCLTOLS_H */

