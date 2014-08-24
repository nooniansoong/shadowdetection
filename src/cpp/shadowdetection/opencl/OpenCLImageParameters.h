#ifndef __OPENCL_IMAGEPARAMETERS_H__ 
#define __OPENCL_IMAGEPARAMETERS_H__

#ifdef _OPENCL

#include "core/opencl/OpenClToolsBase.h"
#include "core/util/Singleton.h"

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
        
        class OpenCLImageParameters : public core::opencl::OpenClBase, public core::util::Singleton<OpenCLImageParameters>{
            friend class core::util::Singleton<OpenCLImageParameters>;
        private:
            cl_mem parametersMem;
            cl_mem originalImageBuffer;
            cl_mem hsvImageBuffer;
            cl_mem hlsImageBuffer;
            
            void createBuffers(const int& numOfPixels, const int parameterCount,
                                const cv::Mat* originalImage, const cv::Mat* hsvImage, 
                                const cv::Mat* hlsImage);
            void setKernelArgs(const cl_uint& numOfParameters, const cl_uint& numOfPixels);
        protected:
            virtual std::string getClassName();
        public:
            OpenCLImageParameters();
            virtual ~OpenCLImageParameters();
            
            virtual void initVars();
            virtual void initWorkVars();
            virtual void cleanUp();
            virtual void cleanWorkPart();
            
            core::util::Matrix<float>* getImageParameters(const cv::Mat* originalImage,
                                                        const cv::Mat* hsvImage,
                                                        const cv::Mat* hlsImage,
                                                        const int& parameterCount);
        };
        
    }
}

#endif

#endif
