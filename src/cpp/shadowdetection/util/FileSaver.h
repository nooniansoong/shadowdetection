#ifndef __FILE_SAVER_H__
#define __FILE_SAVER_H__

#include <string>
#include <fstream>
#include "typedefs.h"
#include "shadowdetection/util/raii/RAIIS.h"

namespace shadowdetection{
    namespace util{
        template<typename T> class FileSaver{
        private:
        protected:
        public:
            static void saveToFile(std::string savePath, T* data, uint dataLength) throw(SDException&){
                std::fstream stream(savePath.c_str(), std::fstream::out | std::fstream::trunc);                
                if (stream.is_open()){
                    shadowdetection::util::raii::FileRaii frai(&stream);
                    for (int i = 0; i < dataLength; i++){
                        if (sizeof(T) != 1)
                            stream << data[i] << std::endl;
                        else
                            stream << (int)data[i] << std::endl;
                    }                   
                }
                else{
                    SDException e(SHADOW_WRITE_UNABLE ,"FileSaver::saveToFile");
                    throw e;
                }
            }
        };
    }
}

#endif