#ifndef __TRAINING_SET_H__ 
#define __TRAINING_SET_H__

#include <string>
#include <vector>
#include "typedefs.h"

namespace shadowdetection{
    namespace learning{
        class TrainingSet{
        private:
            std::string filePath;
            std::vector< KeyVal<std::string> > images;
            
            void readFile() throw (SDException&);
        protected:
        public:
            TrainingSet();
            ~TrainingSet();
            TrainingSet(std::string filePath);
            void setFilePath(std::string filePath);
            void process() throw (SDException&);
            void clear();
        };
    }
}

#endif
