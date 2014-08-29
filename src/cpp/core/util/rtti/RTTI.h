#ifndef __RTTI_H__
#define __RTTI_H__

#include <string>
#include <unordered_map>
#include "core/util/Singleton.h"

#define PREPARE_REGISTRATION(CLASS_NAME) public:\
                                static void* __getClassInstanceSPC(){\
                                    return new CLASS_NAME();\
                                }    
#define REGISTER_CLASS(CLASS_NAME, NAMESPACE)   static core::util::RTTI::RTTI* rtti = core::util::RTTI::RTTI::getInstancePtr();\
                                                std::string clsIDISP = std::string(#NAMESPACE) + std::string("::") + std::string(#CLASS_NAME);\
                                                int a = rtti->registerClassWithInstancer(clsIDISP, &CLASS_NAME::__getClassInstanceSPC);


namespace core{
    namespace util{
        namespace RTTI{
            
            class RTTI : public Singleton<RTTI>{
                friend class Singleton<RTTI>;
            private:
                std::unordered_map<std::string, void*(*)()> mappedInstancers;
            protected:
                RTTI();
                void* getClassInstancePrivate(std::string classID);
            public:
                virtual ~RTTI(){}
                
                template<typename T> T* getClassInstance(std::string classID){
                    void* tmp = getClassInstancePrivate(classID);
                    if (tmp == 0)
                        return 0;
                    T* retPointer = static_cast<T*>(tmp);
                    return retPointer;
                }
                
                int registerClassWithInstancer(std::string classID, void*(*instanceFuinction)());
            };
            
        }
    }
}

#endif
