#include "RTTI.h"

namespace core{
    namespace util{
        namespace RTTI{
            
            using namespace std;
            
            RTTI::RTTI() : Singleton<RTTI>(){
                
            }
            
            void* RTTI::getClassInstancePrivate(string className){
                return 0;
            }
        }
    }
}
