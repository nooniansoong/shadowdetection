#include "ParametersFactory.h" 
#include "core/util/rtti/RTTI.h"

namespace core{
    namespace util{
        
        using namespace core::tools::image;        
        using namespace core::util::RTTI;
        
        IImageParameteres* createImageParameters(){
            core::util::RTTI::RTTI* rtti = core::util::RTTI::RTTI::getInstancePtr();
            IImageParameteres* parametersClass = rtti->getClassInstance<IImageParameteres>("shadowdetection::tools::image::ImageShadowParameters");
            return parametersClass;
        }
        
    }
}