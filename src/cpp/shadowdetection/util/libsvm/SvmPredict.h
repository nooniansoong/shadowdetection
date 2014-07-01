#ifndef __SVM_PREDICT_H__
#define __SVM_PREDICT_H__

#include "shadowdetection/util/Singleton.h"
#include "shadowdetection/util/Matrix.h"
#include <string>

struct svm_model;

namespace shadowdetection{
    namespace util{
        namespace libsvm{
            
            class SvmPredict : public shadowdetection::util::Singleton<SvmPredict> {
                friend class shadowdetection::util::Singleton<SvmPredict>;
            private:
                svm_model* model;
                SvmPredict();
            protected:
            public:
                virtual ~SvmPredict();
                void loadModel(std::string path) throw(SDException&);
                uchar* predict(const shadowdetection::util::Matrix<float>* imagePixelsParameters, const int& pixCount, const int& parameterCount) throw(SDException&);
                bool hasLoadedModel();
            };
            
        }
    }
}

#endif