/* 
 * File:   typedefs.h
 * Author: marko
 *
 * Created on May 31, 2014, 5:40 PM
 */

#ifndef TYPEDEFS_H
#define	TYPEDEFS_H

#include <string>
#include <hash_fun.h>
#include <vector>
#include <sstream>

enum SHADOW_EXCEPTIONS {
    SHADOW_SUCC = 0,
    SHADOW_NO_MEM,
    SHADOW_WRITE_UNABLE,
    SHADOW_READ_UNABLE,
    SHADOW_INVALID_XML,
    SHADOW_NO_OPENCL_PLATFORM,
    SHADOW_NO_OPENCL_DEVICE,
    SHADOW_OUT_OF_BOUNDS,
};

//class sdString : public std::string{
//public:                                                                                          
//    size_t operator()( const sdString& x ) const                                           
//    {                                                                                         
//      return __gnu_cxx::hash<const char*>()(x.c_str());                                              
//    }                                                                                                                                                                                     
//};     

namespace __gnu_cxx {
  template<> struct hash<std::string>
  {
    hash<char*> h;
    size_t operator()(const std::string &s) const
    {
      return h(s.c_str());
    };
  };
}

template<typename T> class KeyVal{
    private:
        T key;
        T val;
    protected:
    public:
        KeyVal(){            
        }
        
        KeyVal(std::string key, std::string val){
            this->key = key;
            this->val = val;
        }
        
        KeyVal (const KeyVal& other){
            key = other.key;
            val = other.val;
        }
        
        virtual ~KeyVal(){            
        }
        
        KeyVal& operator= (KeyVal other){
            key = other.key;
            val = other.val;
        }
        
        T getKey(){
            return key;
        }
        
        T getVal(){
            return val;
        }
};

inline std::string trim(const std::string& input, bool trimCommas = true) {
    std::string whitespaces = " \t\f\v\n\r";
    if (trimCommas == true) {
        whitespaces += ".;,";
    }
    std::string out = input;

    size_t p = out.find_first_not_of(whitespaces);
    out.erase(0, p);

    p = out.find_last_not_of(whitespaces);
    if (std::string::npos != p)
        out.erase(p + 1);

    return out;
}

inline std::vector<std::string> &splitwr(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(trim(s));
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

inline std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    splitwr(s, delim, elems);
    return elems;
}

#endif	/* TYPEDEFS_H */

