#ifndef __PARAMETERS_FACTORY_H__
#define __PARAMETERS_FACTORY_H__

#include "core/tools/image/IImageParameters.h"

namespace core{
    namespace util{
        
        core::tools::image::IImageParameteres* createImageParameters();
        
    }
}

#endif
