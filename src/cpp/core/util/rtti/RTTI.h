#ifndef __RTTI_H__
#define __RTTI_H__

#include <string>
#include <unordered_map>
#include "core/util/Singleton.h"

namespace core{
    namespace util{
        namespace RTTI{
            
            class RTTI : public Singleton<RTTI>{
                friend class Singleton<RTTI>;
            private:
                std::unordered_map<std::string, void*(*)()> mappedInstancers;
            protected:
                RTTI();
                void* getClassInstancePrivate(std::string className);
            public:
                virtual ~RTTI(){}
                
                template<typename T> T* getClassInstance(std::string className){
                    void* tmp = getClassInstancePrivate(className);
                    if (tmp == 0)
                        return 0;
                    T* retPointer = static_cast<T*>(tmp);
                    return retPointer;
                }
            };
            
        }
    }
}

#endif
