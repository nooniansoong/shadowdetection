/* 
 * File:   openCLtols.h
 * Author: marko
 *
 * Created on May 28, 2014, 10:10 PM
 */

#ifndef OPENCLTOLS_H
#define	OPENCLTOLS_H

#ifdef _OPENCL

#include "core/opencl/OpenClToolsBase.h"
#include "core/util/Singleton.h"

struct svm_node;
struct svm_model;

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
            cl_int dummyInt;                                                
            cl_mem inputImage;
            cl_mem hsi1Converted;
            cl_mem hsi2Converted;
            cl_mem tsaiOutput;
            unsigned char* ratios1;
            unsigned char* ratios2;            
            core::util::Matrix<double>* xMatrix;                        
                                    
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
            //======libsvm train section
        private:
            cl_mem clData;
            int clDataLen;
            cl_mem clY;
            cl_mem clX;
            cl_mem clXSquared;            
            bool newTask;
            bool newSelectWorkingSet;
            
            cl_mem clGradDiff;
            cl_mem clObjDiff;
            cl_mem clAlphaStatus;
            cl_mem clYSelectWorkingSet;
            cl_mem clG;
            cl_mem clQD;
            cl_mem clQI;
            
            void createBuffersSVM(  float* data, int dataLen, int i,
                                    char* y, int yLen,
                                    core::util::Matrix<svm_node>* x,
                                    int start, int steps, bool& clDataChanged,
                                    double* xSquared);
            void setKernelArgsSVC( cl_int start, cl_int len, cl_int i, 
                                    cl_int kernel_type, cl_int xW, cl_int dataLen,
                                    cl_double gamma, cl_double coef0, cl_int degree,
                                    bool clDataChanged);
            void setKernelArgsSVR(  cl_int start, cl_int len, cl_int i, 
                                    cl_int kernel_type, cl_int xW, cl_int dataLen,
                                    cl_double gamma, cl_double coef0, cl_int degree,
                                    bool clDataChanged); 
            void createBuffersWorkingSet(const int& activeSize, double* grad_diff,
                                        double* obj_diff, const char* alpha_status, 
                                        const int& l, const char* y, const double* G,
                                        const double* QD, const float* Q_i);
            void setKernelArgsWorkingSet(const int& activeSize, const int& i,
                                        const double& Gmax);
        protected:
        public:
            void get_Q( float* data, int dataLen, int start, int len, int i, int kernel_type, 
                        char* y, int yLen, core::util::Matrix<svm_node>* x, LIBSVM_CLASS_TYPE classType,
                        double gamma, double coef0, int degree, double *xSquared) throw (SDException);
            
            void selectWorkingSet(  const int& activeSize, const int& i, const char* y,
                                    const char* alpha_status, const int& l, double* grad_diff,
                                    const double& Gmax, const double* G, const double* QD, 
                                    const float* Q_i, double* obj_diff) throw (SDException);

            int64_t durrData;
            int64_t durrSetSrgs;
            int64_t durrBuff;
            int64_t durrExec;
            int64_t durrReadBuff;
            //======libsvm predict section
        private:
            void createBuffersPredict(  const core::util::Matrix<float>& parameters, 
                                        svm_model* model);            
            void setKernelArgsPredict(  uint pixelCount, uint paramsPerPixel, 
                                        svm_model* model);
            
            bool            modelChanged;            
            cl_mem          clPixelParameters;
            core::util::Matrix<cl_float>*    modelSVs;
            cl_mem          clModelSVs;
            cl_mem          clModelRHO;
            cl_mem          clModelSVCoefs;
            cl_mem          clModelLabel;
            core::util::Matrix<cl_float>*       svCoefs;
            cl_mem          clModelNsv;
            cl_mem          clPredictResults;
            cl_float*       modelRHOs;
        protected:
        public:
            uchar* predict( svm_model* model, 
                            const core::util::Matrix<float>& parameters);
            void markModelChanged();
        };

    }
}

#endif

#endif	/* OPENCLTOLS_H */

