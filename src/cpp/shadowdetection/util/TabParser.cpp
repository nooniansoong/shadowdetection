#include "TabParser.h"
#include <fstream>

namespace shadowdetection {
    namespace util {
        
        using namespace std;
        
        TabParser::TabParser(){            
        }
        
        TabParser::TabParser(const char* path){
            init(path);
        }
        
        TabParser::~TabParser(){            
        }
        
        void TabParser::init(const char* path) throw (SDException&){
            container.clear();
            ifstream stream(path);
            if (stream.is_open()){
                string line;
                while (getline(stream, line)){
                    line = trim(line);
                    vector<string> tokens = split(line, '\t');
                    if (tokens.size() > 1){
                        string key = tokens[0];
                        string val = tokens[1];
                        KeyVal<string> kv(key, val);
                        container.push_back(kv);
                    }
                }
                stream.close();
            }
            else{
                SDException exc(SHADOW_READ_UNABLE, "Tab file init");
                throw exc;
            }
        }
        
        size_t TabParser::size(){
            return container.size();
        }
        
        KeyVal<std::string> TabParser::get(uint i)throw (SDException&){
            if (i >= size()){
                SDException exc(SHADOW_OUT_OF_BOUNDS, "Get Tab elemenent");
                throw exc;
            }
            return container[i];
        }
        
    }
}
