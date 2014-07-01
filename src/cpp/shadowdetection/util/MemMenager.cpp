#include "MemMenager.h"

namespace shadowdetection{
    namespace util{
        
        using namespace std;
        
        set<void*> MemMenager::allocatedByManager;
        
        void MemMenager::delocate(void* ptr) throw (SDException&) {
            std::set<void*>::iterator iter = allocatedByManager.find(ptr);
            if (allocatedByManager.end() != iter){
                free(ptr);
                allocatedByManager.erase(iter);
            }
            else {
                SDException e(SHADOW_NOT_INITIALIZED_BY_MENAGER_OR_DELETED, "MemMenager::dealocate");
                throw e;
            }
        }
    }
}