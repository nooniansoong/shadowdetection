#include "OpenCLTools.h"
#include "thirdparty/lib_svm/svm.h"
#include "core/util/Timer.h"
#include "core/util/Matrix.h"

#ifdef _OPENCL

namespace shadowdetection {
    namespace opencl {
        
        using namespace std;
        using namespace core::util;                
        
        void OpenclTools::get_Q(float* data, int dataLen, int start, int len, int i, int kernel_type, 
                                char* y, int yLen, Matrix<svm_node>* x, LIBSVM_CLASS_TYPE classType,
                                double gamma, double coef0, int degree, double *xSquared) throw (SDException){
            Timer time;
            
            //bool diffX = xDif(x, xX, xY);
            durrData += time.sinceLastCheck();
            int steps = len - start;
            bool clDataChaned = false;
            createBuffersSVM(   data, dataLen, i, y, yLen, x, start, steps, 
                                clDataChaned, xSquared);
            durrBuff += time.sinceLastCheck();
            size_t local_ws;
            cl_kernel activeKernel;
            if (classType == SVC_Q_TYPE){
                setKernelArgsSVC(   start, len, i, kernel_type, x->getWidth(), dataLen, gamma, 
                                    coef0, degree, clDataChaned);
                local_ws = workGroupSize[3];
                activeKernel = kernel[3];
            }
            else{
                setKernelArgsSVR(   start, len, i, kernel_type, x->getWidth(), dataLen, gamma, 
                                    coef0, degree, clDataChaned);
                local_ws = workGroupSize[4];
                activeKernel = kernel[4];
            }
            durrSetSrgs += time.sinceLastCheck();
            size_t global_ws = shrRoundUp(local_ws, steps);
            err = clEnqueueNDRangeKernel(command_queue, activeKernel, 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "OpenclTools::get_Q clEnqueueNDRangeKernelSVM1");            
            durrExec += time.sinceLastCheck();
            size_t size = steps * sizeof(float);//dataLen
            cl_device_type type;
            err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
            err_check(err, "OpenclTools::get_Q clGetDeviceInfo");
            if (type == CL_DEVICE_TYPE_GPU){
                err = clEnqueueReadBuffer(command_queue, clData, CL_FALSE, start * sizeof(cl_float), size, data + start, 0, NULL, NULL);
                err_check(err, "OpenclTools::get_Q clEnqueueReadBuffer");
            }
            err = clFlush(command_queue);
            err |= clFinish(command_queue);
            err_check(err, "OpenclTools::get_Q clFlush clFinish");
            newTask = false;
            durrReadBuff += time.sinceLastCheck();
        }
        
        Matrix<double>* getValuesFromNodes(Matrix<svm_node>& nodes){
            Matrix<double>* retMat = new Matrix<double>(nodes.getWidth(), nodes.getHeight());
            for (int i = 0; i < nodes.getHeight(); i++){
                for (int j = 0; j < nodes.getWidth(); j++){
                    (*retMat)[i][j] = nodes[i][j].value;
                }
            }
            return retMat;
        }
        
        void OpenclTools::createBuffersSVM( float* data, int dataLen, int i,
                                            char* y, int yLen, Matrix<svm_node>* x, 
                                            int start, int steps, bool& clDataChanged,
                                            double* xSquared){
            bool xChanged = x->changedValues();
            int flag1, flag2;
            cl_device_type type;
            err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
            err_check(err, "OpenclTools::createBuffersSVM clGetDeviceInfo");            
            if (type == CL_DEVICE_TYPE_GPU)
            {                
                flag1 = CL_MEM_WRITE_ONLY;
                flag2 = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;                
            }
            else if (type == CL_DEVICE_TYPE_CPU){
                flag1 = CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR;
                flag2 = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;                
            }
            else{
                SDException exc(SHADOW_NOT_SUPPORTED_DEVICE, "Init buffers, currently not supported device");
                throw exc;
            }
            //====DATA section
            clDataChanged = false;
            if (type == CL_DEVICE_TYPE_GPU){
                if (clData == 0 || (clData != 0 && clDataLen < dataLen)){
                    if (clData != 0){
                        err = clReleaseMemObject(clData);
                        err_check(err, "OpenclTools::createBuffersSVM clReleaseMemObjectCLDATA");
                    }                
                    clData = clCreateBuffer(context, flag1, sizeof(cl_float) * dataLen, 0, &err);
                    err_check(err, "OpenclTools::createBuffersSVM clCreateBufferCLDATA");
                    clDataChanged = true;
                }
                else{              
                    err =  clEnqueueWriteBuffer(command_queue, clData, CL_FALSE, 
                                                start, sizeof(cl_float) * steps, 
                                                (cl_float*)(data + start), 0, 0, 0);
                    err_check(err, "OpenclTools::createBuffersSVM clWriteBufferCLDATA");
                }
            }
            else{
                if (clData != 0){
                    err = clReleaseMemObject(clData);
                    err_check(err, "OpenclTools::createBuffersSVM clReleaseMemObjectCLDATA");
                } 
                clData = clCreateBuffer(context, flag1, sizeof(cl_float) * dataLen, data, &err);
                err_check(err, "OpenclTools::createBuffersSVM clCreateBufferCLDATA");
                clDataChanged = true;
            }            
            clDataLen = dataLen;
            //====
            //====Y section
            //dimension and pointer of y never changes for one task, so can do like this
            if (clY == 0 && y != 0){
                clY = clCreateBuffer(context, flag2, sizeof(cl_char) * yLen, (cl_char*)y, &err);
                err_check(err, "OpenclTools::createBuffersSVM clCreateBufferCLY");
            }
            else if (y != 0){
                if (xChanged){
                    if (type == CL_DEVICE_TYPE_GPU){
                         err = clEnqueueWriteBuffer(command_queue, clY, CL_FALSE, 0, 
                                                    sizeof(cl_char) * yLen, (cl_char*)y, 0, 0, 0);
                         err_check(err, "OpenclTools::createBuffersSVM clEnqueueWriteBufferCLY");
                    }
                }                
            }
            else{
                clY = 0;
            }
            //====
            //====X section
            //dimensions of x never changes for one task, so can do like this
            if (clX == 0 && x != 0){
                size_t size = sizeof(cl_double) * x->getWidth() * x->getHeight();
                if (xMatrix)
                    delete xMatrix;
                xMatrix = getValuesFromNodes(*x);
                clX = clCreateBuffer(context, flag2, size, (cl_double*)xMatrix->getVec(), &err);
                err_check(err, "OpenclTools::createBuffersSVM clCreateBufferCLX");
            }
            else if (x != 0){                
                if (xChanged){
                    if (xMatrix)
                        delete xMatrix;
                    xMatrix = getValuesFromNodes(*x);
                    size_t size = sizeof(cl_double) * x->getWidth() * x->getHeight();
                    err = clEnqueueWriteBuffer( command_queue, clX, CL_FALSE, 
                                                0, size, (cl_double*)xMatrix->getVec(), 0, 0, 0);
                    err_check(err, "OpenclTools::createBuffersSVM clEnqueueWriteBufferCLX");
                }                
            }
            else{
                clX = 0;
            }
            //====
            //==== x squared section
            if (clXSquared == 0 && xSquared != 0){
                size_t size = sizeof(cl_double) * x->getHeight();
                clXSquared = clCreateBuffer(context, flag2, size, (cl_double*)xSquared, &err);
                err_check(err, "OpenclTools::createBuffersSVM clCreateBufferCLXSQUARED");
            }
            else if (xSquared != 0){
                if (type == CL_DEVICE_TYPE_GPU){
                    //if x changed then xsquared is changed too
                    if (xChanged){
                        size_t size = sizeof(cl_double) * x->getHeight();
                        err = clEnqueueWriteBuffer( command_queue, clXSquared, CL_FALSE, 
                                                    0, size, (cl_double*)xSquared, 0, 0, 0);
                        err_check(err, "OpenclTools::createBuffersSVM clEnqueueWriteBufferCLXSQUARED");
                    }
                }
            }
            else{
                clXSquared = 0;
            }
            //====            
        }
        
        void OpenclTools::setKernelArgsSVC( cl_int start, cl_int len, cl_int i, 
                                            cl_int kernel_type, cl_int xW, cl_int dataLen,
                                            cl_double gamma, cl_double coef0, cl_int degree,
                                            bool clDataChanged){
            if (clDataChanged){
                err = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &clData);
                err_check(err, "OpenclTools::setKernelArgsSVC setKernelArgsCLDATA");
            }
            err = clSetKernelArg(kernel[3], 1, sizeof(cl_int), &dataLen);
            err_check(err, "OpenclTools::setKernelArgsSVC clSetKernelArgDATALEN");
            err = clSetKernelArg(kernel[3], 2, sizeof(cl_int), &start);
            err_check(err, "OpenclTools::setKernelArgsSVC clSetKernelArgSTART");
            err = clSetKernelArg(kernel[3], 3, sizeof(cl_int), &len);
            err_check(err, "OpenclTools::setKernelArgsSVC clSetKernelArgLEN");
            err = clSetKernelArg(kernel[3], 4, sizeof(cl_int), &i);
            err_check(err, "OpenclTools::setKernelArgsSVC clSetKernelArgI");
            err = clSetKernelArg(kernel[3], 5, sizeof(cl_int), &kernel_type);
            err_check(err, "OpenclTools::setKernelArgsSVC clSetKernelArgKERNEL_TYPE");
            if (newTask){
                err = clSetKernelArg(kernel[3], 6, sizeof(cl_mem), &clY);
                err_check(err, "OpenclTools::setKernelArgsSVC setKernelArgCLY");
                err = clSetKernelArg(kernel[3], 7, sizeof(cl_mem), &clX);
                err_check(err, "OpenclTools::setKernelArgsSVC setKernelArgCLX");
                err = clSetKernelArg(kernel[3], 12, sizeof(cl_mem), &clXSquared);
                err_check(err, "OpenclTools::setKernelArgsSVC clSetKernelArgCLXSQUARED");                
            }
            err = clSetKernelArg(kernel[3], 8, sizeof(cl_int), &xW);
            err_check(err, "OpenclTools::setKernelArgsSVC clSetKernelArgXW");
            err = clSetKernelArg(kernel[3], 9, sizeof(cl_double), &gamma);
            err_check(err, "OpenclTools::setKernelArgsSVC clSetKernelArgGAMMA");
            err = clSetKernelArg(kernel[3], 10, sizeof(cl_double), &coef0);
            err_check(err, "OpenclTools::setKernelArgsSVC clSetKernelArgCOEF0");
            err = clSetKernelArg(kernel[3], 11, sizeof(cl_int), &degree);
            err_check(err, "OpenclTools::setKernelArgsSVC clSetKernelArgDEGREE");            
        }
        
        void OpenclTools::setKernelArgsSVR( cl_int start, cl_int len, cl_int i, 
                                            cl_int kernel_type, cl_int xW, cl_int dataLen,
                                            cl_double gamma, cl_double coef0, cl_int degree,
                                            bool clDataChanged){
            if (clDataChanged){
                err = clSetKernelArg(kernel[4], 0, sizeof(cl_mem), &clData);
                err_check(err, "OpenclTools::setKernelArgsSVR setKernelArgCLDATA");
            }
            err = clSetKernelArg(kernel[4], 1, sizeof(cl_int), &dataLen);
            err_check(err, "OpenclTools::setKernelArgsSVR clSetKernelArgDATALEN");
            err = clSetKernelArg(kernel[4], 2, sizeof(cl_int), &start);
            err_check(err, "OpenclTools::setKernelArgsSVR clSetKernelArgSTART");
            err = clSetKernelArg(kernel[4], 3, sizeof(cl_int), &len);
            err_check(err, "OpenclTools::setKernelArgsSVR clSetKernelArgLEN");
            err = clSetKernelArg(kernel[4], 4, sizeof(cl_int), &i);
            err_check(err, "OpenclTools::setKernelArgsSVR clSetKernelArgI");
            err = clSetKernelArg(kernel[4], 5, sizeof(cl_int), &kernel_type);
            err_check(err, "OpenclTools::setKernelArgsSVR clSetKernelArgKERNEL_TYPE");
            if (newTask){
                err = clSetKernelArg(kernel[4], 6, sizeof(cl_mem), &clX);
                err_check(err, "OpenclTools::setKernelArgsSVR setKernelArgCLX");
                err = clSetKernelArg(kernel[4], 11, sizeof(cl_mem), &clXSquared);
                err_check(err, "OpenclTools::setKernelArgsSVR clSetKernelArgCLXSQUARED");                
            }
            err = clSetKernelArg(kernel[4], 7, sizeof(cl_int), &xW);
            err_check(err, "OpenclTools::setKernelArgsSVR clSetKernelArgXW");
            err = clSetKernelArg(kernel[4], 8, sizeof(cl_double), &gamma);
            err_check(err, "OpenclTools::setKernelArgsSVR clSetKernelArgGAMMA");
            err = clSetKernelArg(kernel[4], 9, sizeof(cl_double), &coef0);
            err_check(err, "OpenclTools::setKernelArgsSVR clSetKernelArgCOEF0");
            err = clSetKernelArg(kernel[4], 10, sizeof(cl_int), &degree);
            err_check(err, "OpenclTools::setKernelArgsSVR clSetKernelArgDEGREE");            
        }
        
        void OpenclTools::selectWorkingSet( const int& activeSize, const int& i, const char* y,
                                            const char* alpha_status, const int& l, double* grad_diff,
                                            const double& Gmax, const double* G, const double* QD, 
                                            const float* Q_i, double* obj_diff) throw (SDException){
            createBuffersWorkingSet(activeSize, grad_diff, obj_diff, alpha_status, 
                                    l, y, G, QD, Q_i);
            setKernelArgsWorkingSet(activeSize, i, Gmax);
            
            size_t local_ws = workGroupSize[6];
            size_t global_ws = shrRoundUp(local_ws, activeSize);
            err = clEnqueueNDRangeKernel(command_queue, kernel[6], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "clEnqueueNDRangeKernelSELECTWORKINGSET");
            
            cl_device_type type;
            err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
            err_check(err, "OpenclTools::selectWorkingSet clGetDeviceInfo");
            if (type == CL_DEVICE_TYPE_GPU){
                size_t size = sizeof(cl_double) * activeSize;
                err = clEnqueueReadBuffer(command_queue, clGradDiff, CL_FALSE, 0, size, grad_diff, 0, NULL, NULL);
                err_check(err, "OpenclTools::selectWorkingSet clEnqueueReadBufferclGradDiff");
                err = clEnqueueReadBuffer(command_queue, clObjDiff, CL_FALSE, 0, size, obj_diff, 0, NULL, NULL);
                err_check(err, "OpenclTools::selectWorkingSet clEnqueueReadBufferclObjDiff");
            }
            err = clFlush(command_queue);
            err |= clFinish(command_queue);
            err_check(err, "OpenclTools::get_Q clFlush clFinish");
            newSelectWorkingSet = false;
        }
        
        void OpenclTools::createBuffersWorkingSet(  const int& activeSize, double* grad_diff,
                                                    double* obj_diff, const char* alpha_status, 
                                                    const int& l, const char* y, const double* G,
                                                    const double* QD, const float* Q_i){
            int flag1, flag2;
            cl_device_type type;
            err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
            err_check(err, "OpenclTools::createBuffersSVM clGetDeviceInfo");            
            if (type == CL_DEVICE_TYPE_GPU)
            {                
                flag1 = CL_MEM_WRITE_ONLY;
                flag2 = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;                
            }
            else if (type == CL_DEVICE_TYPE_CPU){
                flag1 = CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR;
                flag2 = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;                
            }
            else{
                SDException exc(SHADOW_NOT_SUPPORTED_DEVICE, "Init buffers, currently not supported device");
                throw exc;
            }
            size_t size = sizeof(cl_double) * activeSize;
            if (type == CL_DEVICE_TYPE_GPU){
                clGradDiff = clCreateBuffer(context, flag1, size, 0, &err);
                err_check(err, "OpenclTools::createBuffersWorkingSet clGradDiff");
                clObjDiff = clCreateBuffer(context, flag1, size, 0, &err);
                err_check(err, "OpenclTools::createBuffersWorkingSet clGradDiff");
            }
            else if (type == CL_DEVICE_TYPE_CPU){
                clGradDiff = clCreateBuffer(context, flag1, size, grad_diff, &err);
                err_check(err, "OpenclTools::createBuffersWorkingSet clGradDiff");
                clObjDiff = clCreateBuffer(context, flag1, size, obj_diff, &err);
                err_check(err, "OpenclTools::createBuffersWorkingSet clGradDiff");
            }
            size = sizeof(cl_char) * l;
            if (clAlphaStatus == 0){
                clAlphaStatus = clCreateBuffer(context, flag2, size, (cl_char*)alpha_status, &err);
                err_check(err, "OpenclTools::createBuffersWorkingSet clAlphaStatus");
                clYSelectWorkingSet = clCreateBuffer(context, flag2, size, (cl_char*)y, &err);
                err_check(err, "OpenclTools::createBuffersWorkingSet clYSelectWorkingSet");
                size = sizeof(cl_double) * l;
                clG = clCreateBuffer(context, flag2, size, (cl_double*)G, &err);
                err_check(err, "OpenclTools::createBuffersWorkingSet clG");
            }
            else{
                err = clEnqueueWriteBuffer(command_queue, clAlphaStatus, CL_FALSE, 0, size, alpha_status, 0, 0, 0);
                err_check(err, "OpenclTools::createBuffersWorkingSet clEnqueueWriteBuffer clAlphaStatus");
                err = clEnqueueWriteBuffer(command_queue, clYSelectWorkingSet, CL_FALSE, 0, size, y, 0, 0, 0);
                err_check(err, "OpenclTools::createBuffersWorkingSet clEnqueueWriteBuffer clYSelectWorkingSet");
                size = sizeof(cl_double) * l;
                err = clEnqueueWriteBuffer(command_queue, clG, CL_FALSE, 0, size, G, 0, 0, 0);
                err_check(err, "OpenclTools::createBuffersWorkingSet clEnqueueWriteBuffer clG");
            }
                                    
            size = sizeof(cl_float) * activeSize;
            clQI = clCreateBuffer(context, flag2, size, (cl_float*)Q_i, &err);
            err_check(err, "OpenclTools::createBuffersWorkingSet clQI");
            if (newSelectWorkingSet){                                
                size = sizeof(cl_double) * l;
                clQD = clCreateBuffer(context, flag2, size, (cl_double*)QD, &err);
                err_check(err, "OpenclTools::createBuffersWorkingSet clQD");                
            }
        }
        
        void OpenclTools::setKernelArgsWorkingSet(  const int& activeSize, const int& i,
                                                    const double& Gmax){
            err = clSetKernelArg(kernel[6], 0, sizeof(cl_int), &activeSize);
            err_check(err, "OpenclTools::setKernelArgsWorkingSet setKernelArgACTIVESIZE");
            err = clSetKernelArg(kernel[6], 1, sizeof(cl_int), &i);
            err_check(err, "OpenclTools::setKernelArgsWorkingSet setKernelArgI");            
            if (newSelectWorkingSet){
                err = clSetKernelArg(kernel[6], 2, sizeof(cl_mem), &clYSelectWorkingSet);
                err_check(err, "OpenclTools::setKernelArgsWorkingSet setKernelArgClYSelectWorkingSet");            
                err = clSetKernelArg(kernel[6], 3, sizeof(cl_mem), &clAlphaStatus);
                err_check(err, "OpenclTools::setKernelArgsWorkingSet setKernelArgClAlphaStatus");
            }
            err = clSetKernelArg(kernel[6], 4, sizeof(cl_mem), &clGradDiff);
            err_check(err, "OpenclTools::setKernelArgsWorkingSet setKernelArgClGradDiff");
            err = clSetKernelArg(kernel[6], 5, sizeof(cl_double), &Gmax);
            err_check(err, "OpenclTools::setKernelArgsWorkingSet setKernelArgGmax");
            if (newSelectWorkingSet){
                err = clSetKernelArg(kernel[6], 6, sizeof(cl_mem), &clG);
                err_check(err, "OpenclTools::setKernelArgsWorkingSet setKernelArgClG");
                err = clSetKernelArg(kernel[6], 7, sizeof(cl_mem), &clQD);
                err_check(err, "OpenclTools::setKernelArgsWorkingSet setKernelArgClQD");
            }
            err = clSetKernelArg(kernel[6], 8, sizeof(cl_mem), &clQI);
            err_check(err, "OpenclTools::setKernelArgsWorkingSet setKernelArgClQI");
            err = clSetKernelArg(kernel[6], 9, sizeof(cl_mem), &clObjDiff);
            err_check(err, "OpenclTools::setKernelArgsWorkingSet setKernelArgClObjDiff");
        }
    }        
}

#endif