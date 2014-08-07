/* 
 * File:   typedefs.h
 * Author: marko
 *
 * Created on May 31, 2014, 5:40 PM
 */

#ifndef TYPEDEFS_H
#define	TYPEDEFS_H

#if !defined _OPENCL || defined _AMD
#define _OPENMP_MY
#endif

#include <string>
#ifndef _MAC
#include <hash_fun.h>
#endif
#include <vector>
#include <sstream>
#include <exception>
#include <cstring>
#include <stdint.h>
#include <cstdlib>


enum SHADOW_EXCEPTIONS {
    SHADOW_SUCC = 0,
    SHADOW_NO_MEM,
    SHADOW_WRITE_UNABLE,
    SHADOW_READ_UNABLE,
    SHADOW_INVALID_XML,
    SHADOW_NO_OPENCL_PLATFORM,
    SHADOW_NO_OPENCL_DEVICE,
    SHADOW_OUT_OF_BOUNDS,
    SHADOW_IMAGE_NOT_SUPPORTED_ON_DEVICE,
    SHADOW_NOT_SUPPORTED_DEVICE,
    SHADOW_INVALID_IMAGE_FORMAT,
    SHADOW_DIFFERENT_IMAGES_SIZES,
    SHADOW_NO_MODEL_LOADED,
    SHADOW_INALID_SVM_TYPE,
    SHADOW_INVALID_KERNEL_TYPE,
    SHADOW_CANT_GET_PARAMETERS,
    SHADOW_CANT_PREDICT,
    SHADOW_OPENCL_TOOLS_NOT_INITIALIZED,
    SHADOW_NOT_INITIALIZED_BY_MENAGER_OR_DELETED,
    SHADOW_CANT_ADD_TO_MEM_MENAGER,
    SHADOW_OTHER,
};

static std::string ExceptionStrings[] = {
    "SHADOW_SUCC",
    "SHADOW_NO_MEM",
    "SHADOW_WRITE_UNABLE",
    "SHADOW_READ_UNABLE",
    "SHADOW_INVALID_XML",
    "SHADOW_NO_OPENCL_PLATFORM",
    "SHADOW_NO_OPENCL_DEVICE",
    "SHADOW_OUT_OF_BOUNDS",
    "SHADOW_IMAGE_NOT_SUPPORTED_ON_DEVICE",
    "SHADOW_NOT_SUPPORTED_DEVICE",
    "SHADOW_INVALID_IMAGE_FORMAT",
    "SHADOW_DIFFERENT_IMAGES_SIZES",
    "SHADOW_NO_MODEL_LOADED",
    "SHADOW_INALID_SVM_TYPE",
    "SHADOW_INVALID_KERNEL_TYPE",
    "SHADOW_CANT_GET_PARAMETERS",
    "SHADOW_CANT_PREDICT",
    "SHADOW_OPENCL_TOOLS_NOT_INITIALIZED",
    "SHADOW_NOT_INITIALIZED_BY_MENAGER_OR_DELETED",
    "SHADOW_CANT_ADD_TO_MEM_MENAGER",
    "SHADOW_OTHER"
};

#ifndef _MAC
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
#endif

template<typename T> class KeyVal{
    private:
        T key;
        T val;
    protected:
    public:
        KeyVal(){            
        }
        
        KeyVal(T key, T val){
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
            return *this;
        }
        
        T getKey() const{
            return key;
        }
        
        T getVal() const{
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

class SDException : public std::exception{
private:
    SHADOW_EXCEPTIONS excCode;
    std::string location;
    
    SDException(){}
protected:
public:
    virtual ~SDException() throw(){}
    
    SDException(SHADOW_EXCEPTIONS code, std::string location){
        excCode = code;
        this->location = location;
    }
    virtual const char* what() const throw () {
        std::string retStr = ExceptionStrings[excCode] + " " + location;
        const char* msg = retStr.c_str();
        return msg;
    }
};

template<typename T> inline T maxFunc(T a, T b, T c) {
    T max = a;
    if (b > max)
        max = b;
    if (c > max)
        max = c;

    return max;
}

template<typename T> inline T minFunc(T a, T b, T c) {
    T min = a;
    if (b < min)
        min = b;
    if (c < min)
        min = c;

    return min;
}

inline float radToDegrees(float radians) {
    const float PI_F = 3.14159265358979f;
    float oneRad = 180.f / PI_F;
    return radians * oneRad;
}

template<typename T> inline T clamp(T val, T min, T max) {
    if (val > max)
        return max;
    if (val < min)
        return min;
    return val;
} 

enum LIBSVM_CLASS_TYPE{
    SVC_Q_TYPE,
    SVR_Q_TYPE,
};

#endif	/* TYPEDEFS_H */

