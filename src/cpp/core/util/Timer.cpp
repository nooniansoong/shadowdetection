#include "Timer.h"
#include <time.h>
#include <sys/types.h>
#include <sys/timeb.h>

namespace core{
    namespace util{
        
        Timer::Timer(){
            readCurrent();
            start = current;
            last = current;
        }
        
        Timer::~Timer(){
            
        }
        
        int64_t Timer::sinceLastCheck(){
            readCurrent();
            int64_t diff = current - last;
            last = current;
            return diff;
        }
        
        int64_t Timer::sinceStart(){
            readCurrent();
            int64_t diff = current - start;
            last = current;
            return diff;
        }
        
        void Timer::readCurrent(){
            timeb tb;
            ftime(&tb);
            current = tb.time * 1000 + tb.millitm;
        }
        
    }
}