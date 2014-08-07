#ifndef __REGRESSION_PREDICT_H__
#define __REGRESSION_PREDICT_H__

#include "shadowdetection/util/predicition/IPrediction.h"
#include "shadowdetection/util/Singleton.h"
#include <vector>

namespace shadowdetection{
    namespace util{
        namespace prediction{
            namespace regression{
                
                class RegressionPredict : public IPrediction, public shadowdetection::util::Singleton<RegressionPredict> {
                    friend class shadowdetection::util::Singleton<RegressionPredict>;
                private:
                    bool loadedModel;
                    float borderValue;
                    std::vector<float> coefs;
                    RegressionPredict();
                protected:
                public:
                    virtual ~RegressionPredict();
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
