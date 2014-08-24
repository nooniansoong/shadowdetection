#include "PredictorFactory.h"
#include "core/util/predicition/libsvm/SvmPredict.h"
#include "core/util/predicition/regression/RegressionPredict.h"
#include "core/util/Config.h"

using namespace core::util::prediction;
using namespace core::util::prediction::regression;
using namespace core::util::prediction::svm;
using namespace core::util;
using namespace std;

namespace core{
    namespace util{
        IPrediction* getPredictor(){
            Config* conf = Config::getInstancePtr();
            string type = conf->getPropertyValue("process.Prediction.predictionType");
            if (type == "SVM"){
                return SvmPredict::getInstancePtr();
            }
            else{
                return RegressionPredict::getInstancePtr();
            }
        }
    }
}
