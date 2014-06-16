#include "OpenCLTools.h"
#include "thirdparty/lib_svm/svm.h"

#ifdef _OPENCL

namespace shadowdetection {
    namespace opencl {
        
        using namespace std;
        
        struct cl_svm_node{
            cl_int index;
            cl_double value;
        };               
        
        bool OpenclTools::xDif (const svm_node** x, int xX, int xY){
            bool diffX = false;
            if (x != 0 && xForCl == 0){
                xForCl = new(nothrow) svm_node[xX * xY];
                for (int ix = 0; ix < xX; ix++){
                    memcpy(xForCl + (ix * xY), x[ix], xY * sizeof(svm_node));
                }
            }
            else if (x == 0){
                xForCl = 0;
            }
            else{                
                //different is rare so I think this is better
                for (int ix = 0; ix < xX; ix++){
                    for (int jx = 0; jx < xY; jx++){
                        if (x[ix][jx].index != xForCl[ix * xY + jx].index ||
                                x[ix][jx].value != xForCl[ix * xY + jx].value){
                            diffX = true;
                            break;
                        }
                    }
                    if (diffX)
                        break;
                }
                if (diffX) {                    
                    //h always have same dimension in one training
                    for (int ix = 0; ix < xX; ix++) {
                        memcpy(xForCl + (ix * xY), x[ix], xY * sizeof(svm_node));
                    }
                }
            }
            return diffX;
        }
        
        void OpenclTools::get_Q(float* data, int dataLen, int start, int len, int i, int kernel_type, 
                                char* y, int yLen, const svm_node** x, int xX, int xY, LIBSVM_CLASS_TYPE classType,
                                double gamma, double coef0, int degree, double *x_square) throw (SDException){
            bool diffX = xDif(x, xX, xY);
            
            createBuffersSVM(data, dataLen, y, yLen, xForCl, xX, xY, diffX);
                        
            int steps = len - start;
            size_t local_ws;
            cl_kernel activeKernel;
            if (classType == SVC_Q_TYPE){
                setKernelArgsSVC(start, len, i, kernel_type, xY, dataLen, gamma, coef0, degree);
                local_ws = workGroupSize[3];
                activeKernel = kernel[3];
            }
            else{
                setKernelArgsSVR(start, len, i, kernel_type, xY, dataLen, gamma, coef0, degree);
                local_ws = workGroupSize[4];
                activeKernel = kernel[4];
            }
            size_t global_ws = shrRoundUp(local_ws, steps);
            err = clEnqueueNDRangeKernel(command_queue[1], activeKernel, 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "clEnqueueNDRangeKernelSVM1", -1);            
            size_t size = steps * sizeof(float);//dataLen
            err = clEnqueueReadBuffer(command_queue[1], clData, CL_TRUE, start * sizeof(cl_float), size, data + start, 0, NULL, NULL);
            err_check(err, "clEnqueueReadBufferSVM1", -1);
            clFlush(command_queue[1]);
            clFinish(command_queue[1]);            
        }
        
        void OpenclTools::createBuffersSVM(float* data, int dataLen,
                                           char* y, int yLen,
                                           const svm_node* x, int xX, int xY, bool diffX){
            int flag1, flag2;
            cl_device_type type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
            if (type == CL_DEVICE_TYPE_GPU)
            {                
                flag1 = CL_MEM_WRITE_ONLY;
                flag2 = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;                
            }
            else if (type == CL_DEVICE_TYPE_CPU){
                flag1 = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
                flag2 = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;                
            }
            else{
                SDException exc(SHADOW_NOT_SUPPORTED_DEVICE, "Init buffers, currently not supported device");
                throw exc;
            }            
            if (clData == 0){
                clData = clCreateBuffer(context[1], flag1, sizeof(cl_float) * dataLen, 0, &err);              
            }
            else{
                if (clDataLen >= dataLen){
                   err =  clEnqueueWriteBuffer(command_queue[1], clData, CL_TRUE, 0, sizeof(cl_float) * dataLen, (cl_float*)data, 0, 0, 0);
                   err_check(err, "clEnqueueWriteBuffer", -1);
                }
                else{
                    err = clReleaseMemObject(clData);
                    err_check(err, "clReleaseMemObject", -1);
                    clData = clCreateBuffer(context[1], flag1, sizeof(cl_float) * dataLen, 0, &err);
                }
            }
            clDataLen = dataLen;            
            err_check(err, "clCreateBufferSVM1", -1);
            //dimension of y never changes for one task, so can do like this
            if (clY == 0 && y != 0){
                clY = clCreateBuffer(context[1], flag2, sizeof(cl_char) * yLen, (cl_char*)y, &err);
                err_check(err, "clCreateBufferSVM2", -1);
            }
            else if (y != 0){
                err = clEnqueueWriteBuffer(command_queue[1], clY, CL_TRUE, 0, sizeof(cl_char) * yLen, (cl_char*)y, 0, 0, 0);
                err_check(err, "clEnqueueWriteBuffer1", -1);
            }
            else{
                clY = 0;
            }
            //dimensions of x never changes for one task, so can do like this
            if (clX == 0 && x != 0){
                size_t size = sizeof(cl_svm_node) * xX * xY;
                clX = clCreateBuffer(context[1], flag2, size, (cl_svm_node*)xForCl, &err);
                err_check(err, "clCreateBufferSVM3", -1);
            }
            else if (x != 0){
                if (diffX){
                    size_t size = sizeof(cl_svm_node) * xX * xY;
                    err = clEnqueueWriteBuffer(command_queue[1], clX, CL_TRUE, 0, size, (cl_svm_node*)xForCl, 0, 0, 0);
                    err_check(err, "clEnqueueWriteBuffer2", -1);
                }
            }
            else{
                clX = 0;
            }            
        }
        
        void OpenclTools::setKernelArgsSVC( cl_int start, cl_int len, cl_int i, 
                                            cl_int kernel_type, cl_int xW, cl_int dataLen,
                                            cl_double gamma, cl_double coef0, cl_int degree){
            err = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &clData);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 1, sizeof(cl_int), &dataLen);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 2, sizeof(cl_int), &start);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 3, sizeof(cl_int), &len);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 4, sizeof(cl_int), &i);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 5, sizeof(cl_int), &kernel_type);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 6, sizeof(cl_mem), &clY);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 7, sizeof(cl_mem), &clX);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 8, sizeof(cl_int), &xW);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 9, sizeof(cl_double), &gamma);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 10, sizeof(cl_double), &coef0);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 11, sizeof(cl_int), &degree);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 12, sizeof(cl_mem), 0);
            err_check(err, "clSetKernelArgSVM1", -1);
        }
        
        void OpenclTools::setKernelArgsSVR( cl_int start, cl_int len, cl_int i, 
                                            cl_int kernel_type, cl_int xW, cl_int dataLen,
                                            cl_double gamma, cl_double coef0, cl_int degree){
            err = clSetKernelArg(kernel[4], 0, sizeof(cl_mem), &clData);
            err_check(err, "clSetKernelArgSVM2", -1);
            err = clSetKernelArg(kernel[4], 1, sizeof(cl_int), &dataLen);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[4], 2, sizeof(cl_int), &start);
            err_check(err, "clSetKernelArgSVM2", -1);
            err = clSetKernelArg(kernel[4], 3, sizeof(cl_int), &len);
            err_check(err, "clSetKernelArgSVM2", -1);
            err = clSetKernelArg(kernel[4], 4, sizeof(cl_int), &i);
            err_check(err, "clSetKernelArgSVM2", -1);
            err = clSetKernelArg(kernel[4], 5, sizeof(cl_int), &kernel_type);
            err_check(err, "clSetKernelArgSVM2", -1);            
            err = clSetKernelArg(kernel[4], 6, sizeof(cl_mem), &clX);
            err_check(err, "clSetKernelArgSVM2", -1);
            err = clSetKernelArg(kernel[4], 7, sizeof(cl_int), &xW);
            err_check(err, "clSetKernelArgSVM2", -1);
            err = clSetKernelArg(kernel[3], 8, sizeof(cl_double), &gamma);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 9, sizeof(cl_double), &coef0);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 10, sizeof(cl_int), &degree);
            err_check(err, "clSetKernelArgSVM1", -1);
            err = clSetKernelArg(kernel[3], 11, sizeof(cl_mem), 0);
            err_check(err, "clSetKernelArgSVM1", -1);
        }
        
    }
}

#endif