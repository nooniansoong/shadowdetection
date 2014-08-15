#ifndef __OPENCLTOOLS_TRAIN_H__ 
#define __OPENCLTOOLS_TRAIN_H__

#ifdef _OPENCL

#include "core/opencl/OpenClToolsBase.h"
#include "core/util/Singleton.h"

struct svm_node;

namespace core{
    
    namespace util{
        template<typename T> class Matrix;
    }
    
    namespace opencl{
        namespace libsvm{
            
            class OpenCLToolsTrain : public core::opencl::OpenClBase, public core::util::Singleton<OpenCLToolsTrain>{
                friend class core::util::Singleton<OpenCLToolsTrain>;
            private:
                cl_mem clData;
                int clDataLen;
                cl_mem clY;
                cl_mem clX;
                cl_mem clXSquared;            
                bool newTask;
                bool newSelectWorkingSet;
                core::util::Matrix<double>* xMatrix;

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
                virtual std::string getClassName();
            public:
                OpenCLToolsTrain();
                virtual ~OpenCLToolsTrain();
                
                virtual void initVars();
                virtual void initWorkVars();
                virtual void cleanUp();
                virtual void cleanWorkPart();
                
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
            };
            
        } 
    }
}

#endif
#endif
