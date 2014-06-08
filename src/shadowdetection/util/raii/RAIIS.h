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
            
            /*allocate vector with malloc
             */
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
                        free(vector);
                }                                
            };
            
            /**allocate vector and elements wit malloc*/
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
                            free(matrix[i]);
                        }
                        free(matrix);
                    }
                }
            };
            
        }
    }
}

#endif	/* RAIIS_H */

