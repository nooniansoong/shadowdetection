#include <iostream>
#include "thirdparty/lib_svm/svm.h"
#include "SvmPredict.h"
#ifdef _OPENCL
#include "shadowdetection/opencl/OpenCLTools.h"
#endif
#include "shadowdetection/util/Matrix.h"
#include "shadowdetection/util/Config.h"

namespace shadowdetection {
    namespace util {
        namespace prediction {
            namespace svm {
                using namespace std;
                using namespace shadowdetection::util;
#ifdef _OPENCL
                using namespace shadowdetection::opencl;
#endif

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
                    OpenclTools::getInstancePtr()->markModelChanged();
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
                    ret = MemMenager::allocate<uchar>(pixCount);
                    for (int i = 0; i < pixCount; i++) {
                        if (i % 1000 == 0)
                            cout << "Pix no: " << i << endl;
                        const float* x = (*imagePixelsParameters)[i];
                        svm_node* nodes = MemMenager::allocate<svm_node>(parameterCount + 1);
                        for (int j = 0; j < parameterCount; j++) {
                            nodes[j].index = j + 1;
                            nodes[j].value = x[j];
                        }
                        nodes[parameterCount].index = -1;
                        nodes[parameterCount].value = 0.;
                        double val = svm_predict(model, nodes);
                        ret[i] = (uchar) round(val);
                        MemMenager::delocate(nodes);
                    }
#else
                    if (OpenclTools::getInstancePtr()->hasInitialized() == false) {
                        SDException e(SHADOW_OPENCL_TOOLS_NOT_INITIALIZED, "SvmPredict::predict");
                        throw e;
                    }
                    ret = OpenclTools::getInstancePtr()->predict(model, *imagePixelsParameters);
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