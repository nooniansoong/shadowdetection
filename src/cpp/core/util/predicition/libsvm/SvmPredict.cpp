#include <iostream>
#include <memory>
#include "thirdparty/lib_svm/svm.h"
#include "SvmPredict.h"
#ifdef _OPENCL
#include "core/opencl/libsvm/OpenCLToolsPredict.h"
#endif
#include "core/util/Matrix.h"
#include "core/util/Config.h"
#include "core/util/MemTracker.h"

namespace core{
    namespace util{
        namespace prediction{
            namespace svm {
                using namespace std;
                using namespace core::util;
#ifdef _OPENCL
                using namespace core::opencl::libsvm;
#endif
                
                //REGISTER_SINGLETON(SvmPredict, core::util::prediction::svm)
                
                SvmPredict::SvmPredict() {
                    model = 0;
                }

                SvmPredict::~SvmPredict() {
                    if (model)
                        svm_free_and_destroy_model(&model);
                }

                void SvmPredict::loadModel() throw (SDException&) {
                    string modelFile = Config::getInstancePtr()->getPropertyValue("settings.svm.modelFile");
                    model = svm_load_model(modelFile.c_str());
                    if (model == 0) {
                        SDException e(SHADOW_READ_UNABLE, "SvmPredict::loadModel");
                        throw e;
                    }
#ifdef _OPENCL
                    OpenCLToolsPredict::getInstancePtr()->markModelChanged();
#endif
                }

                uchar* SvmPredict::predict(const Matrix<float>* imagePixelsParameters, const int& pixCount, const int& parameterCount) throw (SDException&) {
                    if (model == 0) {
                        SDException e(SHADOW_NO_MODEL_LOADED, "SvmPredict::predict");
                        throw e;
                    }
                    uchar* ret = 0;
                    if (imagePixelsParameters == 0)
                        return 0;
#ifndef _OPENCL                                
                    ret = New uchar[pixCount];
                    for (int i = 0; i < pixCount; i++) {
                        if (i % 1000 == 0)
                            cout << "Pix no: " << i << endl;
                        const float* x = (*imagePixelsParameters)[i];
                        UNIQUE_PTR(svm_node) nodes(New svm_node[parameterCount + 1]);
                        for (int j = 0; j < parameterCount; j++) {
                            (nodes.get())[j].index = j + 1;
                            (nodes.get())[j].value = x[j];
                        }
                        (nodes.get())[parameterCount].index = -1;
                        (nodes.get())[parameterCount].value = 0.;
                        double val = svm_predict(model, nodes.get());
                        ret[i] = (uchar) round(val);                        
                    }
#else
                    if (OpenCLToolsPredict::getInstancePtr()->hasInitialized() == false) {
                        SDException e(SHADOW_OPENCL_TOOLS_NOT_INITIALIZED, "SvmPredict::predict");
                        throw e;
                    }
                    ret = OpenCLToolsPredict::getInstancePtr()->predict(model, imagePixelsParameters);
#endif
                    return ret;
                }

                bool SvmPredict::hasLoadedModel() {
                    return model != 0;
                }
            }
        }
    }
}