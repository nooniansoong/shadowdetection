#include "RTTI.h"

namespace core{
    namespace util{
        namespace RTTI{
            
            using namespace std;
            
            RTTI::RTTI() : Singleton<RTTI>(){
                
            }
            
            void* RTTI::getClassInstancePrivate(string classID){
                unordered_map<string, void*(*)()>::iterator iter = mappedInstancers.find(classID);
                if (iter == mappedInstancers.end())
                    return 0;
                void*(*instancer)() = iter->second;
                return instancer();
            }
            
            int RTTI::registerClassWithInstancer(string classID, void*(*instanceFuinction)()){
                mappedInstancers[classID] = instanceFuinction;
                return 1;
            }
        }
    }
}
