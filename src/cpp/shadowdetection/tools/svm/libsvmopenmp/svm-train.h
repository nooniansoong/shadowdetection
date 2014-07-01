#ifndef __SVM_TRAIN_H__
#define __SVM_TRAIN_H__

#include "thirdparty/lib_svm/svm.h"

namespace shadowdetection {
        namespace tools {
            namespace svm {
                namespace libsvmopenmp {
                    int train(char* input_file_name, char* model_file_name);
                }
            }
        }
}

#endif