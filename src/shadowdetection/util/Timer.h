#ifndef __TIMER_H__
#define __TIMER_H__

#include "typedefs.h"

namespace shadowdetection{
    namespace util{
        
        class Timer{
        private:
            int64_t start;
            int64_t last;
            int64_t current;
            
            void readCurrent();
        protected:
        public:
            Timer();
            ~Timer();
            int64_t sinceLastCheck();
            int64_t sinceStart();
        };
        
    }
}

#endif