#include "RegressionPredict.h" 
#include "shadowdetection/util/Config.h"

namespace shadowdetection{
    namespace util{
        namespace prediction{
            namespace regression{
                
                using namespace shadowdetection::util;
                using namespace std;
                
                RegressionPredict::RegressionPredict(){
                    loadedModel = false;
                }
                
                RegressionPredict::~RegressionPredict(){
                    
                }
                
                void RegressionPredict::loadModel() throw(SDException&){
                    coefs.clear();
                    Config* conf = Config::getInstancePtr();
                    string numOfArgsStr = conf->getPropertyValue("settings.regression.coefNum");
                    int numOfCoefs = atoi(numOfArgsStr.c_str());
                    for (int i = 0; i < numOfCoefs; i++){
                        string key = "settings.regression.coefNo";
                        char num[10];
                        sprintf(num, "%d", (i + 1));
                        key = key + num;
                        string coefValStr = conf->getPropertyValue(key);
                        float coefVal = atof(coefValStr.c_str());
                        coefs.push_back(coefVal);
                    }
                    string interStrVal = conf->getPropertyValue("settings.regression.Intercept");
                    float intercept = atof(interStrVal.c_str());
                    coefs.push_back(intercept);
                    interStrVal = conf->getPropertyValue("settings.regression.borderValue");
                    borderValue = atof(interStrVal.c_str());
                    loadedModel = true;
                }
                
                uchar* RegressionPredict::predict( const shadowdetection::util::Matrix<float>* imagePixelsParameters, 
                                        const int& pixCount, const int& parameterCount) throw(SDException&){
                    uchar* retArr = MemMenager::allocate<uchar>(pixCount);
                    for (int i = 0; i < pixCount; i++){
                        //intercept
                        float result = coefs[parameterCount];
                        for (int j = 0; j < parameterCount; j++){
                            float a = (*imagePixelsParameters)[i][j] * coefs[j];
                            result += a;
                        }
                        result = -result;
                        result = exp(result);
                        result = 1.f + result;
                        result = 1.f / result;
                        if (result > borderValue)
                            retArr[i] = 1U;
                        else
                            retArr[i] = 0U;
                    }
                    return retArr;
                }
                
                bool RegressionPredict::hasLoadedModel(){
                    return loadedModel;
                }
                
            }
        }
    }
}