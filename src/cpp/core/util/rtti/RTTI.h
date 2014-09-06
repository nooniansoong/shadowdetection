#ifndef __RTTI_H__
#define __RTTI_H__

#include <string>
#include <unordered_map>
#include "core/util/Singleton.h"

namespace core{
    namespace util{
        namespace RTTI{
            
            class RTTI{                
            private:                
            protected:
                void*(*mappedInstancer)();
                bool singleton;
            public:
                RTTI(){};
                virtual ~RTTI(){}                                
                bool isSingleton(){
                    return singleton;
                }
                
                int setSingleton(bool value){
                    singleton = value;
                    return 0;
                }
                
                void* getClassInstance(){
                    return mappedInstancer();
                }
                
                int setInstancer(void*(*instancer)()){
                    mappedInstancer = instancer;
                    return 0;
                }                                
            };
            
        }
    }
}

#endif
