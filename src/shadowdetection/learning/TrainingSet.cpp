#include "TrainingSet.h"
#include <fstream>
#include "shadowdetection/util/raii/RAIIS.h"

namespace shadowdetection{
    namespace learning{
        
        using namespace std;
        using namespace shadowdetection::util::raii;
        
        void TrainingSet::readFile()throw (SDException&){
            fstream file;
            file.open(filePath.c_str(), ifstream::in);
            FileRaii fRaii(&file);
            if (file.is_open()){
                string line;
                while (getline(file, line)){
                    vector<string> tokens = split(line, '\t');
                    if (tokens.size() >= 2){
                        KeyVal<string> pair(tokens[0], tokens[1]);
                        images.push_back(pair);                        
                    }
                }
            }
            else{
                SDException exc(SHADOW_READ_UNABLE, "Learning: Read csv file");
                throw exc;
            }
        }
        
        TrainingSet::TrainingSet(){
            
        }
        
        TrainingSet::~TrainingSet(){
            clear();
        }
        
        TrainingSet::TrainingSet(std::string filePath){
            setFilePath(filePath);
        }
        
        void TrainingSet::setFilePath(std::string filePath){
            this->filePath = filePath;
        }
        
        void TrainingSet::process() throw (SDException&){
            readFile();
        }
        
        void TrainingSet::clear(){
            images.clear();
        }
        
    }
}
