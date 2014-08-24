#ifndef __MEM_MENAGER_H__
#define __MEM_MENAGER_H__

#include <set>
#include <utility>
#include "typedefs.h"

namespace core{
    namespace util{
        
        class MemMenager{            
        private:
            static std::set<void*> allocatedByManager;
        protected:
        public:
            //allocations and dealocations only for vectors
            template<typename T> static T* allocate(int count) throw(SDException&);
            static void delocate(void* ptr) throw (SDException&);
        };
        
        template<typename T> T* MemMenager::allocate(int count) throw(SDException&){
            void* pointer = malloc(count * sizeof (T));
            if (pointer) {
                std::pair < std::set<void*>::iterator, bool> ret = allocatedByManager.insert(pointer);
                if (ret.second == false){
                    SDException exc(SHADOW_CANT_ADD_TO_MEM_MENAGER, "MemMenager::allocate");
                    throw exc;
                }
            }
            return (T*) pointer;
        }
        
    }
}

#endif