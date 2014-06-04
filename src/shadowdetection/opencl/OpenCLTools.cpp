#ifdef _OPENCL
#include "OpenCLTools.h"

#include <iostream>
#include <fstream>

#include "shadowdetection/opencv/OpenCVTools.h"
#include "typedefs.h"

namespace shadowdetection {
    namespace opencl {

        using namespace std;
        using namespace shadowdetection::opencv;
        using namespace cv;

        size_t shrRoundUp(size_t localSize, size_t allSize) {
            if (allSize % localSize == 0) {
                return allSize;
            }
            int coef = allSize / localSize;
            return ((coef + 1) * localSize);
        }

        void OpenclTools::err_check(int err, string err_code) throw (SDException&) {
            if (err != CL_SUCCESS) {
                cout << "Error: " << err_code << "(" << err << ")" << endl;
                if (err == CL_BUILD_PROGRAM_FAILURE) {
                    // Determine the size of the log
                    size_t log_size;
                    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
                    // Allocate memory for the log
                    char *log = (char *) malloc(log_size);
                    // Get the log
                    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                    // Print the log
                    cout << log << endl;
                }
                SDException exc(SHADOW_OTHER, err_code);
                throw exc;
            }
        }
        
        void OpenclTools::initVars(){
//            platform = 0;
//            for (int i = 0; i < MAX_DEVICES; i++) {
//                devices[i] = 0;
//            }
            command_queue = 0;
            program = 0;
            for (int i = 0; i < KERNEL_COUNT; i++) {
                kernel[i] = 0;
            }
            context = 0;
            initWorkVars();
        }
        
        void OpenclTools::initWorkVars(){
            input = 0;
            output1 = 0;
            output2 = 0;
            output3 = 0;
            ratios1 = 0;
            ratios2 = 0;
        }
        
        OpenclTools::OpenclTools() {
            initVars();
        }

        void OpenclTools::cleanWorkPart() {
            if (input)
                clReleaseMemObject(input);
            if (output1)
                clReleaseMemObject(output1);
            if (output2)
                clReleaseMemObject(output2);
            if (output3)
                clReleaseMemObject(output3);            

            if (ratios1)
                delete[] ratios1;
            if (ratios2)
                delete[] ratios2;
            
            initWorkVars();
        }

        void OpenclTools::cleanUp() {
            for (int i = 0; i < KERNEL_COUNT; i++) {
                if (kernel[i] != 0)
                    clReleaseKernel(kernel[i]);
            }
            if (program)
                clReleaseProgram(program);
            if (command_queue)
                clReleaseCommandQueue(command_queue);
            if (context)
                clReleaseContext(context);
            cleanWorkPart();
            
            initVars();
        }

        OpenclTools::~OpenclTools() {
            cleanUp();
        }

        void OpenclTools::init(int platformID, int deviceID, bool listOnly) throw (SDException&) {
            char info[256];
            cl_platform_id platform[MAX_PLATFORMS];
            cl_uint num_platforms;                        
            
            err = clGetPlatformIDs(MAX_PLATFORMS, platform, &num_platforms);
            err_check(err, "clGetPlatformIDs");
            cout << "Found " << num_platforms << " platforms." << endl;                        
            cout << "=============" << endl;
            for (int i = 0; i < num_platforms; i++) {
                cl_device_id devices[MAX_DEVICES];
                cl_uint num_devices;
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 256, info, 0);
                err_check(err, "clGetPlatformInfo");
                cout << "Platform name: " << info << endl;
                try {
#ifdef _AMD
                    err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_CPU, MAX_DEVICES, devices, &num_devices);
#else
                    err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &num_devices);
#endif
                    err_check(err, "clGetDeviceIDs");
                    cout << "Found " << num_devices << " devices" << endl;

                    for (int j = 0; j < num_devices; j++) {
                        err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 256, info, 0);
                        err_check(err, "clGetDeviceInfo");
                        cout << "Device " << j << " name: " << info << endl;
                    }
                }                
                catch (SDException& exception) {
                    cout << "Platform not supported by this build" << endl;
                    cout << exception.what() << endl;
                }
                cout << "=============" << endl;
            }
            
            if (listOnly)
                return;
            
            if (platformID >= num_platforms){
                SDException exc(SHADOW_NO_OPENCL_PLATFORM, "Init platform");
                throw exc;
            }
            
            cl_device_id devices[MAX_DEVICES];
            cl_uint num_devices;
#ifdef _AMD
                err = clGetDeviceIDs(platform[platformID], CL_DEVICE_TYPE_CPU, MAX_DEVICES, devices, &num_devices);
#else
                err = clGetDeviceIDs(platform[platformID], CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &num_devices);
#endif
            err_check(err, "clGetDeviceIDs");
            if (deviceID >= num_devices){
                SDException exc(SHADOW_NO_OPENCL_DEVICE, "Init devices");
                throw exc;
            }
            device = devices[deviceID];

            context = clCreateContext(0, 1, &device, NULL, NULL, &err);
            err_check(err, "clCreateContext");

            cl_bool sup;
            size_t rsize;
            clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof (sup), &sup, &rsize);
            if (sup != CL_TRUE) {
                SDException exception(SHADOW_IMAGE_NOT_SUPPORTED_ON_DEVICE, "Check for image support");
                throw exception;
            }

            command_queue = clCreateCommandQueue(context, device, 0, &err);
            err_check(err, "clCreateCommandQueue");

            ifstream kernelFile;
            kernelFile.open(KERNEL_FILE, ifstream::in | ifstream::binary);
            if (kernelFile.is_open()) {
                char* buffer = 0;
                buffer = new char[MAX_SRC_SIZE];
                if (buffer) {
                    kernelFile.read(buffer, MAX_SRC_SIZE);
                    if (kernelFile.eof()) {
                        size_t readBytes = kernelFile.gcount();
                        program = clCreateProgramWithSource(context, 1, (const char **) &buffer, &readBytes, &err);
                        err_check(err, "clCreateProgramWithSource");

                        err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
                        err_check(err, "clBuildProgram");
                    }
                    delete[] buffer;
                } else {
                    kernelFile.close();
                    SDException exc(SHADOW_NO_MEM, "Init Kernel");
                    throw exc;
                }
                kernelFile.close();
            } else {
                SDException exc(SHADOW_READ_UNABLE, "Init Kernel");
                throw exc;
            }
            return;
        }

        void OpenclTools::createKernels() {
            kernel[0] = clCreateKernel(program, "image_hsi_convert1", &err);
            err_check(err, "clCreateKernel1");
            kernel[1] = clCreateKernel(program, "image_hsi_convert2", &err);
            err_check(err, "clCreateKernel2");
            kernel[2] = clCreateKernel(program, "image_simple_tsai", &err);
            err_check(err, "clCreateKernel3");
        }

        void OpenclTools::createWorkGroupSizes() {
            for (int i = 0; i < KERNEL_COUNT; i++) {
                err = clGetKernelWorkGroupInfo(kernel[i], device, CL_KERNEL_WORK_GROUP_SIZE, sizeof (workGroupSize[i]), &(workGroupSize[i]), NULL);
                err_check(err, "clGetKernelWorkGroupInfo");
            }
        }

        void OpenclTools::createBuffers(uchar* image, u_int32_t height, u_int32_t width, uchar channels) {
            size_t size = width * height * channels;
#ifndef _AMD
            input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, image, &err);
            err_check(err, "clCreateBuffer1");
            output1 = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof (u_int32_t), 0, &err);
            err_check(err, "clCreateBuffer2");
            output2 = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof (u_int32_t), 0, &err);
            err_check(err, "clCreateBuffer3");
            output3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height, 0, &err);
            err_check(err, "clCreateBuffer3");
#else
            int flag = CL_MEM_USE_HOST_PTR;
            input = clCreateBuffer(context, CL_MEM_READ_ONLY | flag, size, image, &err);
            err_check(err, "clCreateBuffer1");            
            output1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof (u_int32_t), 0, &err);
            err_check(err, "clCreateBuffer2");
            output2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof (u_int32_t), 0, &err);
            err_check(err, "clCreateBuffer3");
            output3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, width * height, 0, &err);
            err_check(err, "clCreateBuffer3");
#endif
        }

        Mat* OpenclTools::processRGBImage(uchar* image, u_int32_t width, u_int32_t height, uchar channels) throw (SDException&) {
            if (image == 0) {
                return 0;
            }

            createKernels();
            createWorkGroupSizes();
            createBuffers(image, height, width, channels);

            err = clSetKernelArg(kernel[0], 0, sizeof (cl_mem), &input);
            err_check(err, "clSetKernelArg1");
            err = clSetKernelArg(kernel[0], 1, sizeof (cl_mem), &output1);
            err_check(err, "clSetKernelArg1");
            err = clSetKernelArg(kernel[0], 2, sizeof (u_int32_t), &width);
            err_check(err, "clSetKernelArg1");
            err = clSetKernelArg(kernel[0], 3, sizeof (u_int32_t), &height);
            err_check(err, "clSetKernelArg1");
            err = clSetKernelArg(kernel[0], 4, sizeof (uchar), &channels);
            err_check(err, "clSetKernelArg1");

            err = clSetKernelArg(kernel[1], 0, sizeof (cl_mem), &input);
            err_check(err, "clSetKernelArg2");
            err = clSetKernelArg(kernel[1], 1, sizeof (cl_mem), &output2);
            err_check(err, "clSetKernelArg2");
            err = clSetKernelArg(kernel[1], 2, sizeof (u_int32_t), &width);
            err_check(err, "clSetKernelArg2");
            err = clSetKernelArg(kernel[1], 3, sizeof (u_int32_t), &height);
            err_check(err, "clSetKernelArg2");
            err = clSetKernelArg(kernel[1], 4, sizeof (uchar), &channels);
            err_check(err, "clSetKernelArg2");

            size_t local_ws = workGroupSize[0];
            size_t global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(command_queue, kernel[0], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "clEnqueueNDRangeKernel1");
            clFlush(command_queue);
            clFinish(command_queue);

            err = clSetKernelArg(kernel[2], 0, sizeof (cl_mem), &output1);
            err_check(err, "clSetKernelArg2");
            err = clSetKernelArg(kernel[2], 1, sizeof (cl_mem), &output3);
            err_check(err, "clSetKernelArg2");
            err = clSetKernelArg(kernel[2], 2, sizeof (u_int32_t), &width);
            err_check(err, "clSetKernelArg2");
            err = clSetKernelArg(kernel[2], 3, sizeof (u_int32_t), &height);
            err_check(err, "clSetKernelArg2");
            err = clSetKernelArg(kernel[2], 4, sizeof (uchar), &channels);
            err_check(err, "clSetKernelArg2");

            local_ws = workGroupSize[2];
            global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(command_queue, kernel[2], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "clEnqueueNDRangeKernel3");
            clReleaseMemObject(output1);
            output1 = 0;
            ratios1 = 0;
            ratios1 = new uchar[width * height];
            if (ratios1 == 0) {
                SDException exc(SHADOW_NO_MEM, "Calculate ratios1");
                throw exc;
            }
            err = clEnqueueReadBuffer(command_queue, output3, CL_TRUE, 0, width * height, ratios1, 0, NULL, NULL);
            err_check(err, "clEnqueueReadBuffer1");
            clFlush(command_queue);
            clFinish(command_queue);                       
            
            IplImage* ratiosImage1 = OpenCvTools::get8bitImage(ratios1, height, width);
            IplImage* binarized1 = OpenCvTools::binarize(ratiosImage1);
            delete[] ratios1;
            ratios1 = 0;
            cvReleaseImage(&ratiosImage1);

            local_ws = workGroupSize[1];
            global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(command_queue, kernel[1], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "clEnqueueNDRangeKernel2");
            clReleaseMemObject(input);
            input = 0;
            clFlush(command_queue);
            clFinish(command_queue);

            err = clSetKernelArg(kernel[2], 0, sizeof (cl_mem), &output2);
            err_check(err, "clSetKernelArg2");
            err = clSetKernelArg(kernel[2], 1, sizeof (cl_mem), &output3);
            err_check(err, "clSetKernelArg2");
            err = clSetKernelArg(kernel[2], 2, sizeof (u_int32_t), &width);
            err_check(err, "clSetKernelArg2");
            err = clSetKernelArg(kernel[2], 3, sizeof (u_int32_t), &height);
            err_check(err, "clSetKernelArg2");
            err = clSetKernelArg(kernel[2], 4, sizeof (uchar), &channels);
            err_check(err, "clSetKernelArg2");

            local_ws = workGroupSize[2];
            global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(command_queue, kernel[2], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "clEnqueueNDRangeKernel3");
            clReleaseMemObject(output2);
            output2 = 0;
            ratios2 = 0;
            ratios2 = new uchar[width * height];
            if (ratios2 == 0) {
                SDException exc(SHADOW_NO_MEM, "Calculate ratios2");
                throw exc;
            }
            err = clEnqueueReadBuffer(command_queue, output3, CL_TRUE, 0, width * height, ratios2, 0, NULL, NULL);
            err_check(err, "clEnqueueReadBuffer2");
            clFlush(command_queue);
            clFinish(command_queue);

            clReleaseMemObject(output3);
            output3 = 0;
            clFlush(command_queue);
            clFinish(command_queue);                        
            
            IplImage* ratiosImage2 = OpenCvTools::get8bitImage(ratios2, height, width);
            delete[] ratios2;
            ratios2 = 0;
            IplImage* binarized2 = OpenCvTools::binarize(ratiosImage2);
            cvReleaseImage(&ratiosImage2);

            Mat b1 = cvarrToMat(binarized1, false, false);
            Mat b2 = cvarrToMat(binarized2, false, false);
            Mat* processedImageMat = OpenCvTools::joinTwoOcl(b1, b2);            
            cvReleaseImage(&binarized1);
            cvReleaseImage(&binarized2);
            return processedImageMat;             
        }

    }
}

#endif
