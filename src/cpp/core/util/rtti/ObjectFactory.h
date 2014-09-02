#ifndef __OBJECT_FACTORY_H__
#define __OBJECT_FACTORY_H__

#include "RTTI.h"

namespace core{
    
    namespace tools{
        namespace image{
            class IImageParameteres;
        }
    }
    
    namespace util{
        
        namespace prediction{
            class IPrediction;
        }
        
        namespace RTTI{
            
            class ObjectFactory : public Singleton<ObjectFactory>{
                friend class Singleton<ObjectFactory>;
            private:
            protected:
            public:
                template<typename T> T* createInstance(std::string classID){
                    RTTI* rtti = core::util::RTTI::RTTI::getInstancePtr();
                    T* instance = rtti->getClassInstance<T>(classID);
                    return instance;
                }
                
                core::tools::image::IImageParameteres* createImageParameters();
                core::util::prediction::IPrediction* createPredictor();
            };
            
        }
    }
}

#endif 
