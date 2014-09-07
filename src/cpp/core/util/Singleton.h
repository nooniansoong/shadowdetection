/* 
 * File:   Singleton.h
 * Author: marko
 *
 * Created on May 31, 2014, 6:34 PM
 */

#ifndef SINGLETON_H
#define	SINGLETON_H

#include "core/util/raii/RAIIS.h"
#include "typedefs.h"

namespace core{
    namespace util{
        
        /**
         * template singleton class
         */
        template <class T> class Singleton{
        private:
            static T* instancePtr;
        protected:
            Singleton();            
            virtual ~Singleton();
        public:
            /**
             * 
             * @return instance of T class
             */
            static T* getInstancePtr();
            /**
             * maybe someone will call it sometime !!??
             */
            static void destroy();
        };
        
        template<class T> T* Singleton<T>::instancePtr = 0;
        
        template<class T> Singleton<T>::Singleton(){            
        }
        
        template<class T> Singleton<T>::~Singleton(){            
        }
        
        template<class T> T* Singleton<T>::getInstancePtr(){
            pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
            raii::MutexRaii autoLock(&mutex);
            if (instancePtr == 0){
                instancePtr = New T();
                if (instancePtr == 0){
                    SDException exc(SHADOW_NO_MEM, "Init singleton");
                    throw exc;
                }
            }
            return instancePtr;
        }
        
        template<class T> void Singleton<T>::destroy(){
            if (instancePtr != 0){
                Delete(instancePtr);
            }
            instancePtr = 0;
        }

    }
}

#endif	/* SINGLETON_H */

