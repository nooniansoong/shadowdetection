/* 
 * File:   Config.h
 * Author: marko
 *
 * Created on May 31, 2014, 6:37 PM
 */

#ifndef CONFIG_H
#define	CONFIG_H

#include "Singleton.h"
#include <unordered_map>
#include "typedefs.h"
#include "thirdparty/rapidxml-1.13/rapidxml.hpp"

#define CONFIG_FILE "ShadowDetectionConfig.xml"

namespace core{
    namespace util{        
        
        /**
         * Simple config class, parses XML file and stored mapped values in hahs map
         * Class is Singleton
         */
        class Config : public Singleton<Config>{
            friend class Singleton<Config>;            
        private:
            /**
             * container
             * format is key=xml_node_name.xml_node_name....
             * val = xml_node_value
             */
            std::unordered_map<std::string, std::string> mappedValues;            
            /**
             * process xml file content
             * @param xmlFileContent
             */
            void fillMap(std::string xmlFileContent);
            /**
             * processing in depth from specified xml_node
             * @param node
             * @param currName
             */
            void processNode(rapidxml::xml_node<>* node, std::string currName);                        
            bool rootNodeProcessing;
        protected:
            Config();
            virtual ~Config();
            /**
             * reads xml file and process it
             */
            void init();
        public:
            /**
             * returns value mapped to specified key. Key is in format xml_node_name.xml_node_name....
             * @param key
             * @return 
             */
            virtual std::string getPropertyValue(const std::string& key) throw(SDException&);
        };

    }
}


#endif	/* CONFIG_H */

