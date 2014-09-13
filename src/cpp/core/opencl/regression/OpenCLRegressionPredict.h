#ifndef __OPENCL_REGRESSION_PREDICT_H__
#define __OPENCL_REGRESSION_PREDICT_H__

#include <vector>
#include "core/opencl/OpenClToolsBase.h"
#include "core/util/Singleton.h"
#include "core/util/Matrix.h"

#ifdef _OPENCL
namespace core{
    namespace opencl{
        namespace regression{
            
            class OpenCLRegressionPredict : public OpenClBase, public core::util::Singleton<OpenCLRegressionPredict>{
                friend class core::util::Singleton<OpenCLRegressionPredict>;
            private:
                OpenCLRegressionPredict();
                
                void createBuffers() throw(SDException&);
                void setKernelArgs() throw(SDException&);
                const core::util::Matrix<float>* pixelParameters;
                cl_mem pixelParametersBuff;                
                cl_mem predictedBuff;
                
                uint parameterCount;
                uint pixelCount;                
                
                float* regressionCoefs;
                cl_mem regressionCoefsBuff;
                ulong regressionCoefsNum;
                float borderValue;
            protected:                                
                virtual std::string getClassName();
            public:
                virtual ~OpenCLRegressionPredict();
                
                virtual void initWorkVars();
                virtual void cleanWorkPart();
                virtual void initVars();
                virtual void cleanUp();
                uchar* predict( const core::util::Matrix<float>& imagePixelParameters,
                                const int& pixelCount, const int& parameterCount,
                                std::vector<float> coefs, float borderValue);
            };

        }
    }
}
#endif

#endif
