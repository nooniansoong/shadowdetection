#ifdef _OPENCL
#include "OpenCLTools.h"

#include <iostream>

#include "core/opencv/OpenCV2Tools.h"
#include "core/opencv/OpenCVTools.h"
#include "typedefs.h"
#include "core/util/Config.h"
#include "core/util/raii/RAIIS.h"
#include "thirdparty/lib_svm/svm.h"

#define KERNEL_FILE_1 "image_hci_convert_kernel"
#define KERNEL_FILE_2 "lib_svm"
#define KERNEL_FILE_3 "lib_svm_predict"
#define KERNEL_PATH "src/cpp/shadowdetection/opencl/kernels/"


namespace shadowdetection {
    namespace opencl {

        using namespace std;
        using namespace core::opencv2;
        using namespace core::opencv;
        using namespace cv;
        using namespace core::util;
        using namespace core::util::raii;        

        size_t OpenclTools::shrRoundUp(size_t localSize, size_t allSize) {
            if (allSize % localSize == 0) {
                return allSize;
            }
            int coef = allSize / localSize;
            return ((coef + 1) * localSize);
        }

        void OpenclTools::err_check(int err, string err_code, int programIndex) throw (SDException&) {
            if (err != CL_SUCCESS) {
                cout << "Error: " << err_code << "(" << err << ")" << endl;
                if (err == CL_BUILD_PROGRAM_FAILURE) {
                    // Determine the size of the log
                    size_t log_size;
                    clGetProgramBuildInfo(program[programIndex], device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
                    // Allocate memory for the log
                    char* log = MemMenager::allocate<char>(log_size);
                    VectorRaii vraii(log);
                    // Get the log
                    clGetProgramBuildInfo(program[programIndex], device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                    // Print the log
                    cout << log << endl;
                    
                }
                SDException exc(SHADOW_OTHER, err_code);
                throw exc;
            }
        }
        
        void OpenclTools::initVars(){          
            for (int i = 0; i < PROGRAM_COUNT; i++){
                program[i] = 0;
                context[i] = 0;
                command_queue[i] = 0;
            }
            
            for (int i = 0; i < KERNEL_COUNT; i++) {
                kernel[i] = 0;
            }
            
            //libsvm part
            clY                 = 0;
            clX                 = 0;            
            clData              = 0;
            newTask             = true;
            newSelectWorkingSet = true;
            clXSquared          = 0;
            xMatrix             = 0;
            
            durrData        = 0l;
            durrBuff        = 0l;
            durrExec        = 0l;
            durrReadBuff    = 0l;
            durrSetSrgs     = 0l;            
            
            //training part
            clQD                = 0;
            clAlphaStatus       = 0;
            clYSelectWorkingSet = 0;
            clG                 = 0;
            
            //predict part
            modelSVs        = 0;
            clModelSVs      = 0;
            clModelRHO      = 0;
            clModelSVCoefs  = 0;
            clModelLabel    = 0;
            svCoefs         = 0;
            clModelNsv      = 0;
            modelRHOs       = 0;
            
            initWorkVars();
        }
        
        void OpenclTools::initWorkVars(){
            inputImage = 0;
            hsi1Converted = 0;
            hsi2Converted = 0;
            tsaiOutput = 0;
            ratios1 = 0;
            ratios2 = 0;
            
            //train part            
            clGradDiff = 0;
            clObjDiff = 0;                        
            clQI = 0;            
            
            //predict part
            clPixelParameters = 0;
            clPredictResults = 0;
        }
        
        OpenclTools::OpenclTools() : Singleton<OpenclTools>(){            
            initialized = false;
            modelChanged = true;
            initVars();            
        }

        void OpenclTools::cleanWorkPart() {
            if (inputImage)
                clReleaseMemObject(inputImage);
            if (hsi1Converted)
                clReleaseMemObject(hsi1Converted);
            if (hsi2Converted)
                clReleaseMemObject(hsi2Converted);
            if (tsaiOutput)
                clReleaseMemObject(tsaiOutput);            

            if (ratios1)
                MemMenager::delocate(ratios1);
            if (ratios2)
                MemMenager::delocate(ratios2);
            
            //train part
            if (clGradDiff){
                clReleaseMemObject(clGradDiff);
                err_check(err, "OpenclTools::cleanWorkPart clGradDiff", -1);
            }
            if (clObjDiff){
                clReleaseMemObject(clObjDiff);
                err_check(err, "OpenclTools::cleanWorkPart clObjDiff", -1);
            }
            if (clQI){
                clReleaseMemObject(clQI);
                err_check(err, "OpenclTools::cleanWorkPart clQI", -1);
            }
            
            //predict part
            if (clPixelParameters){
                clReleaseMemObject(clPixelParameters);
                err_check(err, "OpenclTools::cleanWorkPart clPredictResults", -1);
            }
            if (clPredictResults){
                err = clReleaseMemObject(clPredictResults);
                err_check(err, "OpenclTools::cleanWorkPart clPredictResults", -1);
            }    
            initWorkVars();           
        }

        void OpenclTools::cleanUp() {            
            for (int i = 0; i < PROGRAM_COUNT; i++){
                if (program[i])
                    clReleaseProgram(program[i]);
                if (context[i])
                    clReleaseContext(context[i]);
                if (command_queue[i])
                    clReleaseCommandQueue(command_queue[i]);
            }
            
            for (int i = 0; i < KERNEL_COUNT; i++) {
                if (kernel[i]){
                    clReleaseKernel(kernel[i]);
                    err_check(err, "clReleaseKernel", i);
                }
            }
            initialized = false;
            
            //train part
            if (clY){
                err = clReleaseMemObject(clY);
                err_check(err, "clReleaseMemObject2", -1);
            }
                
            if (clX){
                err = clReleaseMemObject(clX);
                err_check(err, "clReleaseMemObject3", -1);
            }
                        
            if (clData){
                err = clReleaseMemObject(clData);
                err_check(err, "clReleaseMemObject1", -1);
            }
            
            if (clXSquared){
                err = clReleaseMemObject(clXSquared);
                err_check(err, "clReleaseMemObjectXSquared", -1);
            }
            
            if (xMatrix){
                delete xMatrix;
            }                        
            
            if (clQD){
                clReleaseMemObject(clQD);
                err_check(err, "OpenclTools::cleanUp clQD", -1);
            }
            
            if (clAlphaStatus){
                clReleaseMemObject(clAlphaStatus);
                err_check(err, "OpenclTools::cleanUp clAlphaStatus", -1);
            }
            
            if (clYSelectWorkingSet){
                clReleaseMemObject(clYSelectWorkingSet);
                err_check(err, "OpenclTools::cleanUp clYSelectWorkingSet", -1);
            }
            
            if (clG){
                clReleaseMemObject(clG);
                err_check(err, "OpenclTools::cleanWorkPart clG", -1);
            } 
            
            durrData = 0l;
            durrBuff = 0l;
            durrExec = 0l;
            durrReadBuff = 0l;
            durrSetSrgs = 0l;
            
            //predict part
            if (modelSVs)
                delete modelSVs;
            if (clModelSVs){
                err = clReleaseMemObject(clModelSVs);
                err_check(err, "OpenclTools::cleanUp clModelSVs", -1);
            }
            if (clModelRHO){
                err = clReleaseMemObject(clModelRHO);
                err_check(err, "OpenclTools::cleanUp clModelRHO", -1);
            }
            if (clModelSVCoefs){
                err = clReleaseMemObject(clModelSVCoefs);
                err_check(err, "OpenclTools::cleanUp clModelSVCoefs", -1);
            }            
            if (clModelLabel){
                err = clReleaseMemObject(clModelLabel);
                err_check(err, "OpenclTools::cleanUp clModelLabel", -1);
            }
            if (svCoefs)
                delete svCoefs;            
            if (clModelNsv){
                err = clReleaseMemObject(clModelNsv);
                err_check(err, "OpenclTools::cleanUp clModelNsv", -1);
            }
            if (modelRHOs)
                MemMenager::delocate(modelRHOs);
            modelChanged = true;
                        
            cleanWorkPart();            
            initVars();            
        }

        OpenclTools::~OpenclTools() {
            cleanUp();
        }
        
        void OpenclTools::loadKernelFile(string& kernelFileName, int index){
            string usePrecompiledStr = Config::getInstancePtr()->getPropertyValue("settings.openCL.UsePrecompiledKernels");
            bool usePrecompiled = usePrecompiledStr.compare("true") == 0;
            if (usePrecompiled){
                bool succ = loadKernelFileFromBinary(kernelFileName, index);
                if (succ){
                    return;
                }
            }
            loadKernelFileFromSource(kernelFileName, index);
            if (usePrecompiled){
                char* fp = saveKernelBinary(kernelFileName, index);
                if (fp != 0){
                    remove(fp);
                    MemMenager::delocate(fp);
                }
            }
        }
        
        void OpenclTools::loadKernelFileFromSource(string& kernelFileName, int index){
            fstream kernelFile;
            string file = KERNEL_PATH + kernelFileName + ".cl";
            kernelFile.open(file.c_str(), ifstream::in);
            FileRaii fRaii(&kernelFile);
            if (kernelFile.is_open()) {
                char* buffer = 0;
                buffer = (char*)MemMenager::allocate<char>(MAX_SRC_SIZE);
                if (buffer) {
                    VectorRaii vraiiBuff(buffer);
                    kernelFile.read(buffer, MAX_SRC_SIZE);
                    if (kernelFile.eof()) {
                        size_t readBytes = kernelFile.gcount();
                        program[index] = clCreateProgramWithSource(context[index], 1, (const char **) &buffer, &readBytes, &err);
                        err_check(err, "clCreateProgramWithSource", index);                                                
                        cout << "Build program: " << kernelFileName << " started" << endl;
                        err = clBuildProgram(program[index], 1, &device, 0, NULL, NULL);
                        err_check(err, "clBuildProgram", index);
                        cout << "Build program: " << kernelFileName << " finished" << endl;
                    }                    
                } else {                    
                    SDException exc(SHADOW_NO_MEM, "Init Kernel");
                    throw exc;
                }                
            } else {
                SDException exc(SHADOW_READ_UNABLE, "Init Kernel");
                throw exc;
            }
        }
        
        bool OpenclTools::loadKernelFileFromBinary(string& kernelFileName, int index){
            fstream kernelFile;
            char deviceName[256];
            err = clGetDeviceInfo(device, CL_DEVICE_NAME, 256, deviceName, 0);
            try{
                err_check(err, "Get device name, load binary kernel", -1);
            }
            catch (SDException& e){
                cout << e.what() << endl;
                return false;
            }
            string file = deviceName;
            file += "_" + kernelFileName + ".ptx";
            kernelFile.open(file.c_str(), ifstream::in | ifstream::binary);
            FileRaii fRaii(&kernelFile);
            if (kernelFile.is_open()) {
                char* buffer = 0;
                buffer = (char*)MemMenager::allocate<char>(MAX_SRC_SIZE);
                if (buffer) {
                    VectorRaii vRaiiBuff(buffer);
                    kernelFile.read(buffer, MAX_SRC_SIZE);
                    if (kernelFile.eof()) {
                        size_t readBytes = kernelFile.gcount();
                        program[index] = clCreateProgramWithBinary(context[index], 1, &device, &readBytes, (const unsigned char**)&buffer, 0, &err);
                        try{
                            err_check(err, "clCreateProgramWithBinary", index);
                        }
                        catch (SDException& e){
                            cout << e.what() << endl;
                            return false;
                        }
                        err = clBuildProgram(program[index], 1, &device, NULL, NULL, NULL);
                        try{
                            err_check(err, "clBuildProgram", index);
                        }
                        catch (SDException& e){
                            cout << e.what() << endl;
                            return false;
                        }
                    }                    
                } else {                    
                    return false;
                }                
            } else {
                return false;
            }
            return true;
        }
        
        char* getCStrCopy(string str){
            char* ret = (char*)MemMenager::allocate<char>(str.size() + 1);
            ret[str.size()] = '\0';
            strcpy(ret, str.c_str());
            return ret;
        }
        
        char* OpenclTools::saveKernelBinary(string& kernelFileName, int index){
            cl_device_type type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
            char deviceName[256];
            err = clGetDeviceInfo(device, CL_DEVICE_NAME, 256, deviceName, 0);
            try{
                err_check(err, "Get device name, save binary kernel", index);
            }
            catch (SDException& e){
                cout << e.what() << endl;
                return 0;
            }
            string file = deviceName;
            file += "_" + kernelFileName + ".ptx";
            fstream kernel;            
            kernel.open(file.c_str(), ofstream::out | ofstream::binary);            
            if (kernel.is_open()){
                FileRaii fRaii(&kernel);
                cl_uint nb_devices;
                size_t retBytes;
                err = clGetProgramInfo(program[index], CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &nb_devices, &retBytes);
                try{
                    err_check(err, "Get num of devices", index);
                }
                catch (SDException& e){
                    cout << e.what() << endl;
                    return getCStrCopy(file);
                }
                
                size_t* binarySize = 0;
                binarySize = MemMenager::allocate<size_t>(nb_devices);
                if (binarySize == 0){
                    SDException exc(SHADOW_NO_MEM, "Get binary sizes");
                    cout << exc.what() << endl;
                    return getCStrCopy(file);
                }
                VectorRaii bsRaii(binarySize);
                err = clGetProgramInfo(program[index], CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * nb_devices, binarySize, 0);
                try{
                    err_check(err, "Get binary size", index);
                }
                catch (SDException& e){
                    cout << e.what() << endl;
                    return getCStrCopy(file);
                }
                uchar**  buffer = 0;
                buffer = MemMenager::allocate<uchar*>(nb_devices);                
                if (buffer != 0){                    
                    for (uint i = 0; i < nb_devices; i++){
                        buffer[i] = MemMenager::allocate<uchar>(binarySize[i]);                        
                    }
                    MatrixRaii mRaii((void**)buffer, nb_devices);
                    
                    size_t read;
                    err = clGetProgramInfo(program[index], CL_PROGRAM_BINARIES, sizeof(unsigned char *)*nb_devices, buffer, &read);
                    try{
                        err_check(err, "Get kernel binaries", index);
                    }
                    catch (SDException& e){
                        cout << e.what() << endl;
                        return getCStrCopy(file);
                    }
                    //because I know that is on one device
                    kernel.write((const char*)buffer[0], binarySize[0]);
                }
                else{
                    SDException exc(SHADOW_NO_MEM, "Save binary kernel");
                    cout << exc.what() << endl;
                    return getCStrCopy(file);
                }                
            }
            else{
                SDException exc(SHADOW_WRITE_UNABLE, "Save binary kernel");
                cout << exc.what() << endl;
                return getCStrCopy(file);
            }
            return 0;
        }
        
        void OpenclTools::init(uint platformID, uint deviceID, bool listOnly) throw (SDException&) {
            char info[256];
            cl_platform_id platform[MAX_PLATFORMS];
            cl_uint num_platforms;                        
            
            err = clGetPlatformIDs(MAX_PLATFORMS, platform, &num_platforms);
            err_check(err, "clGetPlatformIDs", -1);
            cout << "Found " << num_platforms << " platforms." << endl;                        
            cout << "=============" << endl;
            for (uint i = 0; i < num_platforms; i++) {
                cl_device_id devices[MAX_DEVICES];
                cl_uint num_devices;
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 256, info, 0);
                err_check(err, "clGetPlatformInfo", -1);
                cout << "Platform name: " << info << endl;
                try {
#if defined _AMD || defined _MAC
                    err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &num_devices);
#else
                    err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &num_devices);
#endif
                    err_check(err, "clGetDeviceIDs", -1);
                    cout << "Found " << num_devices << " devices" << endl;

                    for (uint j = 0; j < num_devices; j++) {
                        err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 256, info, 0);
                        err_check(err, "clGetDeviceInfo CL_DEVICE_NAME", -1);
                        cl_device_type type;
                        err = clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
                        err_check(err, "clGetDeviceInfo CL_DEVICE_TYPE", -1);
                        string typeStr = "DEVICE_OTHER";
                        if (type == CL_DEVICE_TYPE_CPU)
                            typeStr = "DEVICE_CPU";
                        else if (type == CL_DEVICE_TYPE_GPU)
                            typeStr = "DEVICE_GPU";
                        cl_ulong maxAllocSize;
                        err = clGetDeviceInfo(devices[j],  CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxAllocSize, 0);
                        err_check(err, "clGetDeviceInfo CL_DEVICE_MAX_MEM_ALLOC_SIZE", -1);
                        cout << "Device " << j << " name: " << info << " type: " << typeStr << " max alloc: " << maxAllocSize << endl;
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
#if defined _AMD || defined _MAC
                err = clGetDeviceIDs(platform[platformID], CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &num_devices);
#else
                err = clGetDeviceIDs(platform[platformID], CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &num_devices);
#endif
            err_check(err, "clGetDeviceIDs", -1);
            if (deviceID >= num_devices){
                SDException exc(SHADOW_NO_OPENCL_DEVICE, "Init devices");
                throw exc;
            }
            device = devices[deviceID];
            
            for (int i = 0; i < PROGRAM_COUNT; i++)
            {
                context[i] = clCreateContext(0, 1, &device, NULL, NULL, &err);
                err_check(err, "clCreateContext", -1);
                command_queue[i] = clCreateCommandQueue(context[i], device, 0, &err);
                err_check(err, "clCreateCommandQueue", -1);
            }

            cl_bool sup;
            size_t rsize;
            clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof (sup), &sup, &rsize);
            if (sup != CL_TRUE) {
                SDException exception(SHADOW_IMAGE_NOT_SUPPORTED_ON_DEVICE, "Check for image support");
                throw exception;
            }            
            //image processing section
            string kernelFile = KERNEL_FILE_1;
            loadKernelFile(kernelFile, 0);
            createKernels(0);            
            //svm train section
            kernelFile = KERNEL_FILE_2;
            loadKernelFile(kernelFile, 1);
            createKernels(1);
            //svm predict section
            kernelFile = KERNEL_FILE_3;
            loadKernelFile(kernelFile, 2);
            createKernels(2);
            //create workgroup sizes
            createWorkGroupSizes();
            initialized = true;            
        }

        void OpenclTools::createKernels(int index) {
            if (index == 0){
                kernel[0] = clCreateKernel(program[index], "image_hsi_convert1", &err);
                err_check(err, "clCreateKernel1", index);
                kernel[1] = clCreateKernel(program[index], "image_hsi_convert2", &err);
                err_check(err, "clCreateKernel2", index);
                kernel[2] = clCreateKernel(program[index], "image_simple_tsai", &err);
                err_check(err, "clCreateKernel3", index);
            }
            else if (index == 1){
                kernel[3] = clCreateKernel(program[index], "svcQgetQ", &err);
                err_check(err, "clCreateKernel4", index);
                kernel[4] = clCreateKernel(program[index], "svrQgetQ", &err);
                err_check(err, "clCreateKernel5", index);
                kernel[6] = clCreateKernel(program[index], "selectWorkingSet", &err);
                err_check(err, "clCreateKernel7", index);
            }
            else if (index == 2){
                kernel[5] = clCreateKernel(program[index], "predict", &err);
                err_check(err, "clCreateKernel5", index);
            }
        }

        void OpenclTools::createWorkGroupSizes() {
            for (int i = 0; i < KERNEL_COUNT; i++) {
                if (kernel[i]){
                    err = clGetKernelWorkGroupInfo(kernel[i], device, CL_KERNEL_WORK_GROUP_SIZE, sizeof (workGroupSize[i]), &(workGroupSize[i]), NULL);
                    err_check(err, "clGetKernelWorkGroupInfo", -1);
                }
            }
        }

        void OpenclTools::createBuffers(uchar* image, u_int32_t height, u_int32_t width, uchar channels) {
            size_t size = width * height * channels;
            cl_device_type type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
            if (type == CL_DEVICE_TYPE_GPU)
            {
                inputImage = clCreateBuffer(context[0], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, image, &err);
                err_check(err, "OpenclTools::createBuffers inputImage", -1);
                hsi1Converted = clCreateBuffer(context[0], CL_MEM_READ_WRITE, size * sizeof (u_int32_t), 0, &err);
                err_check(err, "OpenclTools::createBuffers hsi1Converted", -1);
                hsi2Converted = clCreateBuffer(context[0], CL_MEM_READ_WRITE, size * sizeof (u_int32_t), 0, &err);
                err_check(err, "OpenclTools::createBuffers hsi2Converted", -1);
                tsaiOutput = clCreateBuffer(context[0], CL_MEM_WRITE_ONLY, width * height, 0, &err);
                err_check(err, "OpenclTools::createBuffers tsaiOutput", -1);
            }
            else if (type == CL_DEVICE_TYPE_CPU){
                int flag = CL_MEM_USE_HOST_PTR;
                inputImage = clCreateBuffer(context[0], CL_MEM_READ_ONLY | flag, size, image, &err);
                err_check(err, "OpenclTools::createBuffers inputImage", -1);
                hsi1Converted = clCreateBuffer(context[0], CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof (u_int32_t), 0, &err);
                err_check(err, "OpenclTools::createBuffers hsi1Converted", -1);
                hsi2Converted = clCreateBuffer(context[0], CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof (u_int32_t), 0, &err);
                err_check(err, "OpenclTools::createBuffers hsi2Converted", -1);
                tsaiOutput = clCreateBuffer(context[0], CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, width * height, 0, &err);
                err_check(err, "OpenclTools::createBuffers tsaiOutput", -1);
            }
            else{
                SDException exc(SHADOW_NOT_SUPPORTED_DEVICE, 
                                "OpenclTools::createBuffers Init buffers, currently not supported device");
                throw exc;
            }
        }
        
        void OpenclTools::setKernelArgs1(   u_int32_t height, u_int32_t width, 
                                            uchar channels, int lastKernelIndex){
            if (lastKernelIndex >= 0){
                err = clSetKernelArg(kernel[0], 0, sizeof (cl_mem), &inputImage);
                err_check(err, "clSetKernelArg1", -1);
                err = clSetKernelArg(kernel[0], 1, sizeof (cl_mem), &hsi1Converted);
                err_check(err, "clSetKernelArg1", -1);
                err = clSetKernelArg(kernel[0], 2, sizeof (u_int32_t), &width);
                err_check(err, "clSetKernelArg1", -1);
                err = clSetKernelArg(kernel[0], 3, sizeof (u_int32_t), &height);
                err_check(err, "clSetKernelArg1", -1);
                err = clSetKernelArg(kernel[0], 4, sizeof (uchar), &channels);
                err_check(err, "clSetKernelArg1", -1);
                
                if (lastKernelIndex >= 1){
                    err = clSetKernelArg(kernel[1], 0, sizeof (cl_mem), &inputImage);
                    err_check(err, "clSetKernelArg2", -1);
                    err = clSetKernelArg(kernel[1], 1, sizeof (cl_mem), &hsi2Converted);
                    err_check(err, "clSetKernelArg2", -1);
                    err = clSetKernelArg(kernel[1], 2, sizeof (u_int32_t), &width);
                    err_check(err, "clSetKernelArg2",  -1);
                    err = clSetKernelArg(kernel[1], 3, sizeof (u_int32_t), &height);
                    err_check(err, "clSetKernelArg2", -1);
                    err = clSetKernelArg(kernel[1], 4, sizeof (uchar), &channels);
                    err_check(err, "clSetKernelArg2", -1);
                }
            }
        }
        
        void OpenclTools::setKernelArgs2(u_int32_t height, u_int32_t width, unsigned char channels){
            err = clSetKernelArg(kernel[2], 0, sizeof (cl_mem), &hsi1Converted);
            err_check(err, "clSetKernelArg2", -1);
            err = clSetKernelArg(kernel[2], 1, sizeof (cl_mem), &tsaiOutput);
            err_check(err, "clSetKernelArg2", -1);
            err = clSetKernelArg(kernel[2], 2, sizeof (u_int32_t), &width);
            err_check(err, "clSetKernelArg2", -1);
            err = clSetKernelArg(kernel[2], 3, sizeof (u_int32_t), &height);
            err_check(err, "clSetKernelArg2", -1);
            err = clSetKernelArg(kernel[2], 4, sizeof (uchar), &channels);
            err_check(err, "clSetKernelArg2", -1);
        }
        
        void OpenclTools::setKernelArgs3(u_int32_t height, u_int32_t width, unsigned char channels){
            err = clSetKernelArg(kernel[2], 0, sizeof (cl_mem), &hsi2Converted);
            err_check(err, "clSetKernelArg3", -1);
            err = clSetKernelArg(kernel[2], 1, sizeof (cl_mem), &tsaiOutput);
            err_check(err, "clSetKernelArg3",  -1);
            err = clSetKernelArg(kernel[2], 2, sizeof (u_int32_t), &width);
            err_check(err, "clSetKernelArg3", -1);
            err = clSetKernelArg(kernel[2], 3, sizeof (u_int32_t), &height);
            err_check(err, "clSetKernelArg3", -1);
            err = clSetKernelArg(kernel[2], 4, sizeof (uchar), &channels);
            err_check(err, "clSetKernelArg3", -1);
        }
        
        Mat* OpenclTools::processRGBImage(uchar* image, u_int32_t width, u_int32_t height, uchar channels) throw (SDException&) {
            if (image == 0) {
                return 0;
            }
            
            createBuffers(image, height, width, channels);            
            
            setKernelArgs1(height, width, channels, 1);            
            size_t local_ws = workGroupSize[0];
            size_t global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(command_queue[0], kernel[0], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "clEnqueueNDRangeKernel1", -1);
            clFlush(command_queue[0]);
            clFinish(command_queue[0]);
            
            setKernelArgs2(height, width, channels);
            local_ws = workGroupSize[2];
            global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(command_queue[0], kernel[2], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "clEnqueueNDRangeKernel3", -1);
            clReleaseMemObject(hsi1Converted);
            hsi1Converted = 0;
            ratios1 = 0;
            ratios1 = (uchar*)MemMenager::allocate<uchar>(width * height);
            if (ratios1 == 0) {
                SDException exc(SHADOW_NO_MEM, "Calculate ratios1");
                throw exc;
            }
            err = clEnqueueReadBuffer(command_queue[0], tsaiOutput, CL_TRUE, 0, width * height, ratios1, 0, NULL, NULL);
            err_check(err, "clEnqueueReadBuffer1", -1);
            clFlush(command_queue[0]);
            clFinish(command_queue[0]);                       
            
            Mat* ratiosImage1 = OpenCV2Tools::get8bitImage(ratios1, height, width);            
            Mat* binarized1 = OpenCV2Tools::binarize(ratiosImage1);            
            MemMenager::delocate(ratios1);
            ratios1 = 0;
            delete ratiosImage1;

            local_ws = workGroupSize[1];
            global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(command_queue[0], kernel[1], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "clEnqueueNDRangeKernel2", -1);
            clReleaseMemObject(inputImage);
            inputImage = 0;
            clFlush(command_queue[0]);
            clFinish(command_queue[0]);
            
            setKernelArgs3(height, width, channels);
            local_ws = workGroupSize[2];
            global_ws = shrRoundUp(local_ws, width * height);
            err = clEnqueueNDRangeKernel(command_queue[0], kernel[2], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
            err_check(err, "clEnqueueNDRangeKernel3", -1);
            clReleaseMemObject(hsi2Converted);
            hsi2Converted = 0;
            ratios2 = 0;
            ratios2 = (uchar*)MemMenager::allocate<uchar>(width * height);
            if (ratios2 == 0) {
                SDException exc(SHADOW_NO_MEM, "Calculate ratios2");
                throw exc;
            }
            err = clEnqueueReadBuffer(command_queue[0], tsaiOutput, CL_TRUE, 0, width * height, ratios2, 0, NULL, NULL);
            err_check(err, "clEnqueueReadBuffer2", -1);
            clFlush(command_queue[0]);
            clFinish(command_queue[0]);

            clReleaseMemObject(tsaiOutput);
            tsaiOutput = 0;
            clFlush(command_queue[0]);
            clFinish(command_queue[0]);                        
            
            Mat* ratiosImage2 = OpenCV2Tools::get8bitImage(ratios2, height, width);            
            Mat* binarized2 = OpenCV2Tools::binarize(ratiosImage2);            
            MemMenager::delocate(ratios2);
            ratios2 = 0;
            delete ratiosImage2;
            
            Mat* processedImageMat = OpenCV2Tools::joinTwoOcl(*binarized1, *binarized2);
            delete binarized1;
            delete binarized2;                        
            return processedImageMat;             
        }
        
        bool OpenclTools::hasInitialized(){
            return initialized;
        }

    }
}

#endif
