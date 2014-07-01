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
#include <fstream>
#include "opencv2/core/core.hpp"
#include "shadowdetection/util/MemMenager.h"


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
            
            class ImageNewRaii{
            private:
                cv::Mat* image;
                ImageNewRaii(){
                    image = 0;
                }
            protected:
            public:
                ImageNewRaii(cv::Mat* img){
                    image = img;
                }
                ~ImageNewRaii(){
                    if (image != 0)
                        delete image;
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
            
            class FileRaii{
            private:
                std::fstream* file;
                FileRaii(){}
            protected:
            public:
                FileRaii(std::fstream* file){
                    this->file = file;
                }
                
                ~FileRaii(){
                    if (file){
                        file->flush();
                        file->close();
                    }
                }
            };
            
            /**allocate vector with MemMenager*/
            class VectorRaii{
            private:
                VectorRaii(){
                    vector = 0;
                }
                
                void* vector;                
            protected:
            public:                                
                VectorRaii(void* vec){
                    vector = vec;
                }
                
                ~VectorRaii(){
                    if (vector)                        
                        shadowdetection::util::MemMenager::delocate(vector);
                }                                
            };
            
            /**allocate vector and elements with MemMenager*/
            class MatrixRaii{
            private:
                MatrixRaii(){
                    matrix = 0;
                }
                
                void** matrix;
                int dim;
            protected:
            public:
                MatrixRaii(void** mat, int dimension){
                    matrix = mat;
                    dim = dimension;
                }
                
                ~MatrixRaii(){
                    if (matrix != 0){
                        for (int i = 0; i < dim; i++){                            
                            shadowdetection::util::MemMenager::delocate(matrix[i]);
                        }
                        shadowdetection::util::MemMenager::delocate(matrix);
                    }
                }
            };
            
            template <typename T> class PointerRaii{
            private:                                
                T* pointer;                
            protected:
            public:
                PointerRaii(){
                    pointer = 0;
                }
                
                PointerRaii(T* pt){
                    pointer = pt;
                }
                
                ~PointerRaii(){
                    if (pointer)                        
                        delete pointer;
                }
                
                void setPointer(T* pt){
                    if (pointer)
                        delete pointer;
                    pointer = pt;
                }
                
                void deactivate(){
                    pointer = 0;
                }
            };
            
        }
    }
}

#endif	/* RAIIS_H */

