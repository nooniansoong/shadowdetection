#ifndef __SVM_PREDICT_H__
#define __SVM_PREDICT_H__

#include "core/util/Singleton.h"
#include "core/util/predicition/IPrediction.h"

struct svm_model;

namespace core{
    namespace util{
        namespace prediction{
            namespace svm{

                class SvmPredict : public IPrediction, public core::util::Singleton<SvmPredict> {
                    friend class core::util::Singleton<SvmPredict>;
                private:
                    svm_model* model;
                    SvmPredict();
                protected:
                public:
                    virtual ~SvmPredict();
                    virtual void loadModel() throw(SDException&);
                    virtual uchar* predict( const core::util::Matrix<float>* imagePixelsParameters, 
                                            const int& pixCount, const int& parameterCount) throw(SDException&);
                    virtual bool hasLoadedModel();
                };

            }
        }
    }
}

#endif