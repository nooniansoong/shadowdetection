#ifndef __TRAINING_SET_H__ 
#define __TRAINING_SET_H__

#include <string>
#include <vector>
#include <fstream>
#include "typedefs.h"

namespace shadowdetection{
    namespace learning{
        class TrainingSet{
        private:
            std::string filePath;
            std::vector< KeyVal<std::string> > images;
            
            void readFile() throw (SDException&);
            void processImages(std::string output) throw (SDException&);
            float** processImage(std::string orImage, std::string maskImage, int& rowDimesion, int& pixelNum);
        protected:
        public:
            TrainingSet();
            ~TrainingSet();
            TrainingSet(std::string filePath);
            void setFilePath(std::string filePath);
            void process(std::string output) throw (SDException&);
            void clear();
        };
    }
}

#endif
