#ifndef __PREDICTOR_MANAGER_H__ 
#define __PREDICTOR_MANAGER_H__

#include "shadowdetection/util/predicition/IPrediction.h"

namespace shadowdetection{
    namespace util{
        shadowdetection::util::prediction::IPrediction* getPredictor();
    }
}

#endif
