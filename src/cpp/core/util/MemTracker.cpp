#ifdef _DEBUG

#include "MemTracker.h"
#include <utility>

namespace core{
    namespace util{
        
        using namespace std;
        
        bool MemTrackerStruct::operator==(const MemTrackerStruct &other) const{
            size_t a = reinterpret_cast<size_t>(ptr);
            size_t b = reinterpret_cast<size_t>(other.ptr);
            return a == b;
        }

        bool MemTrackerStruct::operator<(const MemTrackerStruct& rhs) const{
            size_t a = reinterpret_cast<size_t>(ptr);
            size_t b = reinterpret_cast<size_t>(rhs.ptr);
            if (a == b){
                int g = 0;
                ++g;
            }
            bool retRes = a < b;
            return retRes;
        }
        
        MemTrackerStruct& MemTrackerStruct::operator=(MemTrackerStruct other){
            ptr = other.ptr;
            lineNum = other.lineNum;
            file = other.file;
            return *this;
        }

        std::string MemTrackerStruct::toString(){
            char buffer[6];                
            std::string ret = file;
            sprintf(buffer, "%d", lineNum);
            ret += buffer;
            return ret;
        }
        
        set<MemTrackerStruct> MemTracker::allocatedByManager;
        
        void MemTracker::add(MemTrackerStruct ptr) throw(SDException&){
            pair< set<MemTrackerStruct>::iterator, bool > succ = allocatedByManager.insert(ptr);
            if (succ.second == false){
                SDException exc(SHADOW_CANT_ADD_TO_MEM_MENAGER, "MemTracker::add");
                throw exc;
            }
        }
        
        void MemTracker::remove(void* ptr) throw(SDException&){
            MemTrackerStruct tmp;
            tmp.ptr = ptr;
            set<MemTrackerStruct>::iterator iter = allocatedByManager.find(tmp);
            if (iter != allocatedByManager.end()){
                allocatedByManager.erase(iter);
                return;
            }
            SDException exc(SHADOW_NOT_INITIALIZED_BY_MENAGER_OR_DELETED, "MemTracker::remove");
            throw exc;
        }
        
        void MemTracker::remove(const void* ptr) throw(SDException&){
            void* tmpPtr = const_cast<void*>(ptr);
            remove(tmpPtr);
        }
        
        string MemTracker::getUnfreed(){
            string retString = "Unfreed:\n";
            set<MemTrackerStruct>::iterator iter = allocatedByManager.begin();
            while (iter != allocatedByManager.end()){
                MemTrackerStruct mtStruct = *iter;
                retString += mtStruct.toString() + "\n";
                iter++;
            }
            return retString;
        }
    }
}

#endif