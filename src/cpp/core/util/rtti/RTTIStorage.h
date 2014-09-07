#ifndef __RTTI_STORAGE_H__
#define __RTTI_STORAGE_H__

#include <unordered_map>
#include "core/util/Singleton.h"

namespace core{
    namespace util{
        namespace RTTI{
            
            class RTTIStorage : public core::util::Singleton<RTTIStorage>{
                friend class core::util::Singleton<RTTIStorage>;
            private:
                pthread_mutex_t mutex;
                std::unordered_map<std::string, RTTI*> mappedRTTIs;
            protected:
                RTTIStorage() : core::util::Singleton<RTTIStorage>(){
                    mutex = PTHREAD_MUTEX_INITIALIZER;
                }
            public:
                virtual ~RTTIStorage(){}
                
                int registerClass(std::string classID, RTTI* rtti){                
                    raii::MutexRaii autoLock(&mutex);
                    mappedRTTIs[classID] = rtti;
                    return 1;
                }
                
                RTTI* getClassRTTI(std::string classID){
                    raii::MutexRaii autoLock(&mutex);
                    std::unordered_map<std::string, RTTI*>::iterator iter = mappedRTTIs.find(classID);
                    if (iter == mappedRTTIs.end())
                        return 0;
                    return iter->second;
                }
            };
            
        }
    }
}

#endif
