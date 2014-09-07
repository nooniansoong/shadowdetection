#ifndef __IPROCESSOR_H__ 
#define __IPROCESSOR_H__

#include "typedefs.h"

namespace core{
    namespace process{
        
        class IProcessor{
        private:
        protected:
            IProcessor(){}  
        public:
            virtual ~IProcessor(){}
            virtual void init() throw (SDException&) = 0;
            virtual void process(int argc, char **argv) = 0;
        };
        
    }
}

#endif
