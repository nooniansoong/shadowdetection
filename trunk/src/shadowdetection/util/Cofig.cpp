#include "Config.h"
#include <fstream>
#include <cstring>

//#define ARRAY_KEY_TAG "array"
//#define TYPE_KEY_TAG "type"
//#define NAME_KEY_TAG "name"

namespace shadowdetection {
    namespace util{
        using namespace std;
        using namespace __gnu_cxx;
        using namespace rapidxml;
        
        Config::Config() : Singleton<Config>(){
            rootNodeProcessing = true;
            init();            
        }
        
        Config::~Config(){
            
        }
        
        string Config::getPropertyValue(const string& key){
            hash_map<string, string>::iterator iter = mappedValues.find(key);
            if (iter != mappedValues.end()){
                return iter->second;
            }
            else 
                return "";
        }
        void Config::init() {            
            string path = CONFIG_FILE;
            ifstream inputFile;
            inputFile.open(path.c_str(), ifstream::in);
            if (inputFile && inputFile.is_open()) {
                ostringstream stream;
                string line;
                while (getline(inputFile, line)) {
                    stream << line;
                }
                inputFile.close();
                string xmlContent = stream.str();
                fillMap(xmlContent);
            } else {
                throw (int)SHADOW_READ_UNABLE;
            }
        }
        void Config::fillMap(string xmlFileContent) {
            try {
                xml_document<> doc;
                const char* constContent = xmlFileContent.c_str();
                char* content = new char[strlen(constContent) + 1];
                strcpy(content, constContent);
                doc.parse<0>(content);
                string currName = "";
                for (xml_node<>* root = doc.first_node(); root; root = root->next_sibling()) {                    
                    processNode(root, currName);
                }
            } catch (exception exc) {
                throw (int) SHADOW_INVALID_XML;
            }
        }

        string getNodeName(xml_node<>* node) {
            string retString = node->name();
            return retString;
        }        
        
        void Config::processNode(xml_node<>* node, string currName) {
            string refName = currName;
            if (refName != "") {
                refName += ".";
            }

            if (rootNodeProcessing) {
                rootNodeProcessing = false;
            } else {
                string nodeName = getNodeName(node);
                refName += nodeName;
            }            
            
            bool hasChildren = false;            
            for (xml_node<>* child = node->first_node(); child; child = child->next_sibling()) {
                hasChildren = true;                                
                processNode(child, refName);
            }

            if (hasChildren == false) {
                string nodeValStr = node->value();
                nodeValStr = trim(nodeValStr);
                refName = trim(refName);
                mappedValues[refName] = nodeValStr;                               
            }
        }
        
    }
}