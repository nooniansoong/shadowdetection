/* 
 * File:   Config.h
 * Author: marko
 *
 * Created on May 31, 2014, 6:37 PM
 */

#ifndef CONFIG_H
#define	CONFIG_H

#include "Singleton.h"
#include <hash_map>
#include "typedefs.h"
#include "thirdparty/rapidxml-1.13/rapidxml.hpp"

#define CONFIG_FILE "ShadowDetectionConfig.xml"

namespace shadowdetection {
    namespace util{
        
//        enum ATTRIB_TYPE{
//                ATTRIB_TYPE_NO_TYPE,
//                ATTRIB_TYPE_TYPE_NUM,
//                ATTRIB_TYPE_TYPE_STR,
//        };
        
        class Config : public Singleton<Config>{
            friend class Singleton<Config>;            
        private:            
            __gnu_cxx::hash_map<std::string, std::string> mappedValues;            
            void fillMap(std::string xmlFileContent);
            void processNode(rapidxml::xml_node<>* node, std::string currName);                        
            bool rootNodeProcessing;
        protected:
            Config();
            virtual ~Config();
            void init();
        public:
            virtual std::string getPropertyValue(const std::string& key);
        };

    }
}


#endif	/* CONFIG_H */

