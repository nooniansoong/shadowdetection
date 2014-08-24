#include "ParametersFactory.h" 
#include "shadowdetection/tools/image/ImageShadowParameters.h"

namespace core{
    namespace util{
        
        using namespace core::tools::image;
        using namespace shadowdetection::tools::image;
        
        IImageParameteres* createImageParameters(){
            return new ImageShadowParameters();
        }
        
    }
}