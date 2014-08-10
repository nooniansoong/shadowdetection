#include "PredictorFactory.h"
#include "shadowdetection/util/predicition/libsvm/SvmPredict.h"
#include "shadowdetection/util/predicition/regression/RegressionPredict.h"
#include "shadowdetection/util/Config.h"

using namespace shadowdetection::util::prediction;
using namespace shadowdetection::util::prediction::regression;
using namespace shadowdetection::util::prediction::svm;
using namespace std;

namespace shadowdetection{
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
