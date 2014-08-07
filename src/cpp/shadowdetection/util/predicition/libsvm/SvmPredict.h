#ifndef __SVM_PREDICT_H__
#define __SVM_PREDICT_H__

#include "shadowdetection/util/Singleton.h"
#include "shadowdetection/util/predicition/IPrediction.h"

struct svm_model;

namespace shadowdetection{
    namespace util{
        namespace prediction{
            namespace svm{

                class SvmPredict : public IPrediction, public shadowdetection::util::Singleton<SvmPredict> {
                    friend class shadowdetection::util::Singleton<SvmPredict>;
                private:
                    svm_model* model;
                    SvmPredict();
                protected:
                public:
                    virtual ~SvmPredict();
                    virtual void loadModel() throw(SDException&);
                    virtual uchar* predict( const shadowdetection::util::Matrix<float>* imagePixelsParameters, 
                                            const int& pixCount, const int& parameterCount) throw(SDException&);
                    virtual bool hasLoadedModel();
                };

            }
        }
    }
}

#endif