#ifndef __MATRIX_H__
#define __MATRIX_H__

namespace shadowdetection{
    namespace util{
        
        template <typename T> class Matrix{
        private:
            T* vector;
            int width, height;
            bool changed;
            Matrix();
        protected:
        public:
            Matrix(T* const* mat, int width, int height);
            virtual ~Matrix();
            void swap(int i, int j);
            const bool changedValues();
            //operators                                   
            const T* operator[](int idx) const{
                return (vector + (idx * width));
            }
            
            operator const T*() const{
                return vector;
            }
            
            int getWidth() const{
                return width;
            }
            
            int getHeight() const{
                return height;
            }
        };
        
        template <typename T> Matrix<T>::Matrix(){
            vector = 0;
            changed = true;
        }
        
        template <typename T> Matrix<T>::Matrix(T* const* mat, int width, int height){
            this->width = width;
            this->height = height;
            vector = new T[height * width];
            for (int i = 0; i < height; i++)
                memcpy(vector + (i * width), mat[i], width * sizeof(T));
            changed = true;
        }
        
        template <typename T> Matrix<T>::~Matrix(){
            if (vector)
                delete[] vector;
        }
        
        template <typename T> void Matrix<T>::swap(int i, int j){
            T* tmp = new T[width];
            memcpy(tmp, vector + (i * width), width * sizeof(T));
            memcpy(vector + (i * width), vector + (j * width), width * sizeof(T));
            memcpy(vector + (j * width), tmp, width * sizeof(T));
            delete[] tmp;
            changed = true;
        }
        
        template <typename T> const bool Matrix<T>::changedValues(){
            bool val = changed;
            changed = false;
            return val;
        }
        
    }
}

#endif
