#include "OpenCLTools.h"
#include "thirdparty/lib_svm/svm.h"
//#include "shadowdetection/util/MemMenager.h"

#ifdef _OPENCL
namespace shadowdetection {
    namespace opencl {
        
        using namespace shadowdetection::util;
        
        size_t getSVsWidth(svm_model* model){
            size_t maxWidth = 0;
            for (int i = 0; i < model->l; i++){
                size_t currWidth = 1;
                svm_node* node = model->SV[i];
                while (node->index != -1){
                    currWidth++;
                    node++;
                }
                if (maxWidth < currWidth)
                    maxWidth = currWidth;
            }
            return maxWidth;
        }
        
        cl_svm_node* convertSVs(svm_model* model, size_t& svsWidth){
            int height = model->l;
            int width = getSVsWidth(model);
            svsWidth = width;
            Matrix<cl_svm_node> matrix(width, height);
            for (int i = 0; i < height; i++){
                for (int j = 0; j < width; j++){
                    matrix[i][j].index = model->SV[i][j].index;
                    matrix[i][j].value = model->SV[i][j].value;
                    if (model->SV[i][j].index == -1)
                        break;
                }
            }
            const cl_svm_node* vec = matrix;
            cl_svm_node* retVec = MemMenager::allocate<cl_svm_node>(width * height);
            memcpy(retVec, vec, width * height * sizeof(cl_svm_node));
            return retVec;
        }
        
        cl_double* convertSVCoefs(svm_model* model){
            int width = model->l;
            int height = model->nr_class - 1;
            Matrix<cl_double> matrix(model->sv_coef, width, height);
            cl_double* retVec = MemMenager::allocate<cl_double>(width * height);
            const cl_double* vec = matrix;
            memcpy(retVec, vec, width * height * sizeof(cl_double));
            return retVec;
        }
        
        size_t getDecValuesSize(svm_model* model){
            if (model->param.svm_type == ONE_CLASS || model->param.svm_type == EPSILON_SVR ||
                model->param.svm_type == NU_SVR)
                return 1;
            else{
                return model->nr_class * (model->nr_class - 1) / 2;
            }            
        }
        
        void OpenclTools::createBuffersPredict( const Matrix<svm_node>& parameters, 
                                                svm_model* model){
            cl_device_type type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
            int flag1, flag2;
            if (type == CL_DEVICE_TYPE_GPU)
            {                
                flag1 = CL_MEM_WRITE_ONLY;
                flag2 = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;                
            }
            else if (type == CL_DEVICE_TYPE_CPU){
                flag1 = CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
                flag2 = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;                
            }
            else{
                SDException exc(SHADOW_NOT_SUPPORTED_DEVICE, "Init buffers, currently not supported device");
                throw exc;
            }
            
            size_t size = parameters.getWidth() * parameters.getHeight() * sizeof(cl_svm_node);
            cl_svm_node* pNodes = (cl_svm_node*)parameters.getVec();
            clPixelParameters = clCreateBuffer( context[2], flag2, size, 
                                                pNodes, &err);
            err_check(err, "OpenclTools::createBuffersPredict clPixelParameters", -1);            
            
            if (modelChanged){
                if (modelSVs != 0)
                    MemMenager::delocate(modelSVs);
                modelSVs = convertSVs(model, modelSvsWidth);
                size = modelSvsWidth * model->l * sizeof(cl_svm_node);
                if (clModelSVs){
                    err = clReleaseMemObject(clModelSVs);
                    err_check(err, "OpenclTools::createBuffersPredict delete clModelSVs", -1);
                }    
                clModelSVs = clCreateBuffer(context[2], flag2, size, (cl_svm_node*)modelSVs, &err);
                err_check(err, "OpenclTools::createBuffersPredict clModelSVs", -1);            
                
                if (svCoefs)
                    MemMenager::delocate(svCoefs);
                svCoefs = convertSVCoefs(model);
                size = (model->nr_class - 1) * (model->l) * sizeof(cl_double);
                    if (clModelSVCoefs){
                    err = clReleaseMemObject(clModelSVCoefs);
                    err_check(err, "OpenclTools::createBuffersPredict delete clModelSVCoefs", -1);
                }
                clModelSVCoefs = clCreateBuffer(context[2], flag2, size, (cl_double*)svCoefs, &err);
                err_check(err, "OpenclTools::createBuffersPredict clModelSVCoefs", -1);            

                size = (model->nr_class * (model->nr_class - 1) / 2) * sizeof(cl_double);
                if (clModelRHO){
                    err = clReleaseMemObject(clModelRHO);
                    err_check(err, "OpenclTools::createBuffersPredict delete clModelRHO", -1);
                }
                clModelRHO = clCreateBuffer(context[2], flag2, size, (cl_double*)model->rho, &err);
                err_check(err, "OpenclTools::createBuffersPredict clModelRHO", -1);            

                size = model->nr_class * sizeof(cl_int);
                if (clModelLabel){
                    err = clReleaseMemObject(clModelLabel);
                    err_check(err, "OpenclTools::createBuffersPredict delete clModelLabel", -1);
                }
                clModelLabel = clCreateBuffer(context[2], flag2, size, (cl_int*)model->label, &err);
                err_check(err, "OpenclTools::createBuffersPredict clModelLabel", -1);            

                size = model->nr_class * sizeof(cl_int);
                if (clModelNsv){
                    err = clReleaseMemObject(clModelNsv);
                    err_check(err, "OpenclTools::createBuffersPredict delete clModelNsv", -1);
                }
                clModelNsv = clCreateBuffer(context[2], flag2, size, (cl_int*)model->nSV, &err);
                err_check(err, "OpenclTools::createBuffersPredict clModelNsv", -1);
                
                modelChanged = false;
            }            
            
            size = parameters.getHeight() * sizeof(cl_uchar);
            clPredictResults = clCreateBuffer(context[2], flag1, size, 0, &err);
            err_check(err, "OpenclTools::createBuffersPredict clPredictResults", -1);            
        }                
        
        void OpenclTools::setKernelArgsPredict( uint pixelCount, uint paramsPerPixel, 
                                                svm_model* model){
            err = clSetKernelArg(kernel[5], 0, sizeof(cl_mem), &clPixelParameters);
            err_check(err, "OpenclTools::setKernelArgsPredict clPixelParameters", -1);
            err = clSetKernelArg(kernel[5], 1, sizeof(cl_uint), &pixelCount);
            err_check(err, "OpenclTools::setKernelArgsPredict pixelCount", -1);
            err = clSetKernelArg(kernel[5], 2, sizeof(cl_uint), &paramsPerPixel);
            err_check(err, "OpenclTools::setKernelArgsPredict paramsPerPixel", -1);
            err = clSetKernelArg(kernel[5], 3, sizeof(cl_int), &model->nr_class);
            err_check(err, "OpenclTools::setKernelArgsPredict nr_class", -1);
            err = clSetKernelArg(kernel[5], 4, sizeof(cl_int), &model->l);
            err_check(err, "OpenclTools::setKernelArgsPredict l", -1);            
            err = clSetKernelArg(kernel[5], 5, sizeof(cl_int), &modelSvsWidth);
            err_check(err, "OpenclTools::setKernelArgsPredict svsWidth", -1);
            err = clSetKernelArg(kernel[5], 6, sizeof(cl_mem), &clModelSVs);
            err_check(err, "OpenclTools::setKernelArgsPredict clModelSVs", -1);
            err = clSetKernelArg(kernel[5], 7, sizeof(cl_mem), &clModelSVCoefs);
            err_check(err, "OpenclTools::setKernelArgsPredict clModelSVCoefs", -1);
            err = clSetKernelArg(kernel[5], 8, sizeof(cl_mem), &clModelRHO);
            err_check(err, "OpenclTools::setKernelArgsPredict clModelRHO", -1);
            err = clSetKernelArg(kernel[5], 9, sizeof(cl_mem), &clModelLabel);
            err_check(err, "OpenclTools::setKernelArgsPredict clModelLabel", -1);
            err = clSetKernelArg(kernel[5], 10, sizeof(cl_mem), &clModelNsv);
            err_check(err, "OpenclTools::setKernelArgsPredict clModelNsv", -1);
            err = clSetKernelArg(kernel[5], 11, sizeof(cl_int), &model->free_sv);
            err_check(err, "OpenclTools::setKernelArgsPredict free_sv", -1);
            err = clSetKernelArg(kernel[5], 12, sizeof(cl_int), &model->param.svm_type);
            err_check(err, "OpenclTools::setKernelArgsPredict param.svm_type", -1);
            err = clSetKernelArg(kernel[5], 13, sizeof(cl_int), &model->param.kernel_type);
            err_check(err, "OpenclTools::setKernelArgsPredict param.kernel_type", -1);
            err = clSetKernelArg(kernel[5], 14, sizeof(cl_int), &model->param.degree);
            err_check(err, "OpenclTools::setKernelArgsPredict param.degree", -1);
            err = clSetKernelArg(kernel[5], 15, sizeof(cl_double), &model->param.gamma);
            err_check(err, "OpenclTools::setKernelArgsPredict param.gamma", -1);
            err = clSetKernelArg(kernel[5], 16, sizeof(cl_double), &model->param.coef0);
            err_check(err, "OpenclTools::setKernelArgsPredict param.coef0", -1);
            //====
            err = clSetKernelArg(kernel[5], 17, sizeof(cl_mem), &clPredictResults);
            err_check(err, "OpenclTools::setKernelArgsPredict clPredictResults", -1);
            //====
            size_t size = model->nr_class * sizeof(cl_int) * workGroupSize[5];
            err = clSetKernelArg(kernel[5], 18, size, 0);            
            err_check(err, "OpenclTools::setKernelArgsPredict start", -1);
                        
            size = model->nr_class * sizeof(cl_int) * workGroupSize[5];
            err = clSetKernelArg(kernel[5], 19, size, 0);
            err_check(err, "OpenclTools::setKernelArgsPredict vote", -1);            
        }
        
        uchar* OpenclTools::predict(svm_model* model, const Matrix<svm_node>& parameters){            
            createBuffersPredict(parameters, model);            
            setKernelArgsPredict(parameters.getHeight(), parameters.getWidth(), model);
            
            size_t local_ws = workGroupSize[5];
            int numValues = parameters.getHeight() * parameters.getWidth();
            size_t global_ws = shrRoundUp(local_ws, numValues);
            err = clEnqueueNDRangeKernel(command_queue[2], kernel[5], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "OpenclTools::predict clEnqueueNDRangeKernel", -1);
            size_t size = parameters.getHeight() * sizeof(cl_uchar);
            uchar* retVec = MemMenager::allocate<uchar>(numValues);
            err = clEnqueueReadBuffer(command_queue[2], clPredictResults, CL_TRUE, 0, size, retVec, 0, NULL, NULL);
            err_check(err, "OpenclTools::predict clEnqueueReadBuffer", -1);
            clFlush(command_queue[2]);
            clFinish(command_queue[2]);
            return retVec;
        }
        
        void OpenclTools::markModelChanged(){
            modelChanged = true;
        }
        
    }
}
#endif