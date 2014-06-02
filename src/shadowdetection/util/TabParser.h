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

        class TabParser {
        private:
            std::vector< KeyVal<std::string> > container;
        protected:
        public:
            TabParser();
            TabParser(const char* path);
            virtual ~TabParser();
            void init(const char* path) throw (SDException&);
            size_t size();
            KeyVal<std::string> get(int i) throw (SDException&);
        };

    }
}

#endif	/* TABPARSER_H */

