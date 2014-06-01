/* 
 * File:   RAIIS.h
 * Author: marko
 *
 * Created on June 1, 2014, 8:42 PM
 */

#ifndef RAIIS_H
#define	RAIIS_H

#include <cv.h>
#include <highgui.h>
#include <pthread.h>

namespace shadowdetection {
    namespace util{
        namespace raii{
            
            class ImageRaii{
            private:
                IplImage* image;
                ImageRaii(){
                    image = 0;
                }
            protected:
            public:
                ImageRaii(IplImage* img){
                    image = img;
                }
                ~ImageRaii(){
                    if (image != 0)
                        cvReleaseImage(&image);
                }
            };
            
            class MutexRaii{
            private:
                pthread_mutex_t* mutex;
                MutexRaii(){
                    mutex = 0;
                }
            protected:
            public:
                MutexRaii(pthread_mutex_t* mtx){
                    mutex = mtx;
                    pthread_mutex_lock(mutex);
                }
                
                ~MutexRaii(){
                    if (mutex != 0)
                        pthread_mutex_unlock(mutex);
                }
            };
            
        }
    }
}

#endif	/* RAIIS_H */

