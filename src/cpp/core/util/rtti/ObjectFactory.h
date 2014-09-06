#ifndef __OBJECT_FACTORY_H__
#define __OBJECT_FACTORY_H__

#include "RTTI.h"
#include "RTTIStorage.h"


#define PREPARE_REGISTRATION(CLASS_NAME) public:\
                                static void* __getClassInstanceSPC(){\
                                    return New CLASS_NAME();\
                                }    

#define REGISTER_CLASS(CLASS_NAME, NAMESPACE)   core::util::RTTI::RTTI* rtti = new core::util::RTTI::RTTI();\
                                                int a = rtti->setSingleton(false);\
                                                int b = rtti->setInstancer(&CLASS_NAME::__getClassInstanceSPC);\
                                                std::string clsIDISP = std::string(#NAMESPACE) + std::string("::") + std::string(#CLASS_NAME);\
                                                int c = core::util::RTTI::RTTIStorage::getInstancePtr()->registerClass(clsIDISP, rtti);

#define REGISTER_SINGLETON(CLASS_NAME, NAMESPACE)   core::util::RTTI::RTTI* rtti = new core::util::RTTI::RTTI();\
                                                    int a = rtti->setSingleton(true);\
                                                    int b = rtti->setInstancer(&CLASS_NAME::__getClassInstanceSPC);\
                                                    std::string clsIDISP = std::string(#NAMESPACE) + std::string("::") + std::string(#CLASS_NAME);\
                                                    int c = core::util::RTTI::RTTIStorage::getInstancePtr()->registerClass(clsIDISP, rtti);\
                                                    NAMESPACE::CLASS_NAME* singleton = NAMESPACE::CLASS_NAME::getInstancePtr();\
                                                    int d = core::util::RTTI::ObjectFactory::getInstancePtr()->registerSingleton(clsIDISP, singleton);

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
                ObjectFactory(){}
                
                std::unordered_map<std::string, void*> mappedSingletons;
                void* getSingleton(std::string classID);
            protected:
            public:
                virtual ~ObjectFactory(){}
                
                template<typename T> T* createInstance(std::string classID) throw (SDException&){
                    core::util::RTTI::RTTI* rtti = RTTIStorage::getInstancePtr()->getClassRTTI(classID);
                    if (rtti != 0){
                        if (rtti->isSingleton() == false){
                            T* instance = static_cast<T*>(rtti->getClassInstance());
                            if (instance == 0){
                                SDException exc(SHADOW_CLASS_NOT_REGISTRETED, "Class not registreted, class ID: " + classID);
                                throw exc;
                            }
                            return instance;
                        }
                        else{
                            void* tmpInstance = getSingleton(classID);
                            if (tmpInstance != 0){
                                T* instance = static_cast<T*>(tmpInstance);
                                return instance;                                
                            }
                            SDException exc(SHADOW_CLASS_NOT_REGISTRETED, "Class not registreted, class ID: " + classID);
                            throw exc;
                        }
                    }
                    SDException exc(SHADOW_CLASS_NOT_REGISTRETED, "Class not registreted, class ID: " + classID);
                    throw exc;
                }
                
                core::tools::image::IImageParameteres* createImageParameters();
                core::util::prediction::IPrediction* createPredictor();
                int registerSingleton(std::string classID, void* singleton);
            };
            
        }
    }
}

#endif 
