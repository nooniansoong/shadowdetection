#ifndef __OPENCLTOOLS_PREDICT_H__ 
#define __OPENCLTOOLS_PREDICT_H__

#ifdef _OPENCL

#include "core/opencl/OpenClToolsBase.h"
#include "core/util/Singleton.h"

struct svm_model;

namespace core{
    
    namespace util{
        template<typename T> class Matrix;
    }
    
    namespace opencl{
        namespace libsvm{
            
            class OpenCLToolsPredict : public core::opencl::OpenClBase, public core::util::Singleton<OpenCLToolsPredict>{
                friend class core::util::Singleton<OpenCLToolsPredict>;
            private:
                cl_int          dummyInt;
                
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
                
                void createBuffers(const core::util::Matrix<float>* parameters, svm_model* model);            
                void setKernelArgs(uint pixelCount, uint paramsPerPixel, svm_model* model);                
            protected:
                virtual std::string getClassName();
            public:
                OpenCLToolsPredict();
                virtual ~OpenCLToolsPredict();
                
                virtual void initVars();
                virtual void initWorkVars();
                virtual void cleanUp();
                virtual void cleanWorkPart();
                uchar* predict( svm_model* model, const core::util::Matrix<float>* parameters);
                void markModelChanged();
            };
            
        }
    }
}

#endif

#endif
