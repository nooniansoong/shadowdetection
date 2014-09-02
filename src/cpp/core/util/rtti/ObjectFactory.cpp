#include "ObjectFactory.h"
#include "core/tools/image/IImageParameters.h"
#include "core/util/predicition/IPrediction.h"
#include "core/util/predicition/libsvm/SvmPredict.h"
#include "core/util/predicition/regression/RegressionPredict.h"
#include "core/util/Config.h"

namespace core {
    namespace util {
        namespace RTTI {

            using namespace core::tools::image;
            using namespace core::util::prediction;
            using namespace std;
            using namespace core::util;
            
            IImageParameteres* ObjectFactory::createImageParameters() {
                string classID = Config::getInstancePtr()->getPropertyValue("general.Prediction.parametersClass");
                core::util::RTTI::RTTI* rtti = core::util::RTTI::RTTI::getInstancePtr();
                IImageParameteres* parametersClass = rtti->getClassInstance<IImageParameteres>(classID);
                return parametersClass;
            }
            
            IPrediction* ObjectFactory::createPredictor(){
                Config* conf = Config::getInstancePtr();
                string type = conf->getPropertyValue("general.Prediction.predictionType");
                if (type == "SVM"){
                    return svm::SvmPredict::getInstancePtr();
                }
                else{
                    return regression::RegressionPredict::getInstancePtr();
                }
            }
        }
    }
}
