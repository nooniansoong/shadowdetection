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
#include <functional>
#include <vector>
#include <sstream>
#include <exception>
#include <cstring>
#include <stdint.h>
#include <cstdlib>
#include <cmath>

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
    SHADOW_NULL_POINTER,
    SHADOW_NOT_FOUND_PROPERTY,
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
    "SHADOW_NULL_POINTER",
    "SHADOW_NOT_FOUND_PROPERTY",
    "SHADOW_OTHER"
};

template<typename T> class Pair{
    private:        
    protected:
        T first;
        T second;
    public:
        Pair(){            
        }
        
        Pair(T first, T second){
            this->first = first;
            this->second = second;
        }
        
        Pair(const Pair& other){
            first = other.first;
            second = other.second;
        }
        
        virtual ~Pair(){            
        }
        
        Pair& operator=(Pair other){
            first = other.first;
            second = other.second;
            return *this;
        }
        
        T getFirst(){
            return first;
        }
        
        T getSecond(){
            return second;
        }
        
        const T& getFirst() const{
            return first;
        }
        
        const T& getSecond() const{
            return second;
        }
        
        bool operator==(const Pair &other) const{
            return (first == other.first && second == other.second);
        }                
        
        operator size_t() const{
            size_t hash = 7U;
            hash = 97U * hash + first;
            hash = 97U * hash + second;
            return hash;
        }
};

template<typename T> class Triple : public Pair<T>{
    private:
        T third;
    protected:
    public:
        Triple(){
        }
        
        Triple(T first, T second, T third) : Pair<T>(first, second){
            this->third = third;            
        }
        
        Triple(const Triple& other) : Pair<T>(other.first, other.second){
            third = other.third;
        }
        
        virtual ~Triple(){            
        }
        
        T getThird(){
            return third;
        }
        
        const T& getThird() const{
            return third;
        }
};

namespace std {
    
    template<typename T> struct hash< Pair<T> > {

        size_t operator()(const Pair<T> &kv) const {
            size_t retVal = kv;
            return retVal;
        };
    };

    template<typename T> struct hash< Pair<T>* > {

        size_t operator()(const Pair<T>* kv) const {
            size_t retVal = *kv;
            return retVal;
        };
    };

    struct eqKeyVal{
        template<typename T> bool operator()(Pair<T>* kv1, Pair<T>* kv2) const {
            return (kv1->getFirst() == kv2->getFirst() && kv1->getSecond() == kv2->getSecond());
        }
    };

}

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
    SDException(){        
    }
protected:
public:
    virtual ~SDException() throw(){}
    
    SDException(SHADOW_EXCEPTIONS code, std::string location){
        excCode = code;
        this->location = location;        
    }
    
    virtual const char* what() const throw () {
        std::string retStr = ExceptionStrings[excCode] + " " + location;
        char* ret = new char[retStr.length() + 1];
        strcpy(ret, retStr.c_str());
        return ret;
    }
    
    std::string handleException(){
        const char* err = this->what();
        std::string ret(err);
        delete err;
        return ret;
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

template<typename T> inline T inEpsilonRange(T val, const T& reperVal, const T& epsilon) {
    if (abs(val - reperVal) <= epsilon)
        return true;
    return false;
}

enum LIBSVM_CLASS_TYPE{
    SVC_Q_TYPE,
    SVR_Q_TYPE,
};

#endif	/* TYPEDEFS_H */


    
