#ifndef __REGRESSION_PREDICT_H__
#define __REGRESSION_PREDICT_H__

#include "core/util/predicition/IPrediction.h"
#include "core/util/Singleton.h"
#include <vector>

namespace core{
    namespace util{
        namespace prediction{
            namespace regression{
                
                class RegressionPredict : public IPrediction, public core::util::Singleton<RegressionPredict> {
                    friend class core::util::Singleton<RegressionPredict>;
                private:
                    bool loadedModel;
                    float borderValue;
                    std::vector<float> coefs;
                    RegressionPredict();
                protected:
                public:
                    virtual ~RegressionPredict();
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
