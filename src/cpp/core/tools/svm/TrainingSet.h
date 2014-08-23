#ifndef __TRAINING_SET_H__ 
#define __TRAINING_SET_H__

#include <fstream>
#include "typedefs.h"
#include "core/util/Matrix.h"

namespace core{
    namespace tools{
        namespace svm{
            class TrainingSet{
            private:
                std::string filePath;
                std::vector< Pair<std::string> > images;

                void readFile() throw (SDException&);
                void processImages(std::string output, bool outputAll) throw (SDException&);
                core::util::Matrix<float>* processImage(std::string orImage, std::string maskImage, int& rowDimesion, int& pixelNum);
            protected:
            public:
                TrainingSet();
                ~TrainingSet();
                TrainingSet(std::string filePath);
                void setFilePath(std::string filePath);
                void process(std::string output, bool processAll) throw (SDException&);
                void clear();
            };
        }
    }
}

#endif
