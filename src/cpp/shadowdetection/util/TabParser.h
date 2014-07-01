/* 
 * File:   TabParser.h
 * Author: marko
 *
 * Created on June 1, 2014, 7:03 PM
 */

#ifndef TABPARSER_H
#define	TABPARSER_H

#include "typedefs.h"

namespace shadowdetection {
    namespace util {

        /**
         * class for handling files with tab separated values, each record in different line
         */
        class TabParser {
        private:
            /**
             * container
             */
            std::vector< KeyVal<std::string> > container;
        protected:
        public:
            TabParser();
            TabParser(const char* path);
            virtual ~TabParser();
            /**
             * reads tab file and fill container with value pairs
             * @param path
             */
            void init(const char* path) throw (SDException&);
            /**
             * cast
             * @return 
             * number of pairs in container
             */
            size_t size();
            /**         
             * @param i
             * @return 
             * pair at position i
             */
            KeyVal<std::string> get(int i) throw (SDException&);
        };

    }
}

#endif	/* TABPARSER_H */

