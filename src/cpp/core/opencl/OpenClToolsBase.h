#ifndef __OPENCL_BASE_H__
#define __OPENCL_BASE_H__

#ifdef _OPENCL

#ifndef _MAC
#include <CL/cl.h>
#else
#include <OpenCL/opencl.h>
#endif
#include "typedefs.h"
#include <vector>

namespace core{
    namespace opencl{
        
        struct cl_svm_node{
            cl_int index;
            cl_double value;
        };
        
        struct cl_svm_node_float{
            cl_int index;
            cl_float value;
        };
        
        class OpenClBase{
        private:
            std::string dirToOpenclprogramFiles;
            
            std::string getBinaryFile(const std::string& programFileName);
            /**
             * calculate work group sizes for each kernel
             */
            void createWorkGroupSizes();
            /**
             * create openCL kernels from program
             */
            void createKernels();
            /**
             * get kernel function names used in class from config
             * @return 
             */
            virtual std::vector<std::string> getKernelNamesForClass();
            virtual std::vector<std::string> getProgramFilesForClass();
        protected:
            cl_device_id device;
            cl_int err;
            //program connected variables
            cl_command_queue command_queue;
            cl_program program;
            cl_context context;
            //kernel connected variables
            cl_kernel* kernel;
            size_t* workGroupSize;
            int kernelCount;
            //class variables
            bool initialized;
            
            /**
             * check for openCL error
             * @param err
             * @param err_code
             */
            void err_check(int err, std::string err_code) throw (SDException&);
            /**
             * uses for get number of global work size
             * @param localSize
             * @param allSize
             * @return 
             */
            size_t shrRoundUp(size_t localSize, size_t allSize);
            /**
             * global function for load program
             * @param kernelFileName
             */
            void loadProgramFile(const std::string& programFileName);
            /**
             * load program from source
             * @param kernelFileName
             */
            void loadProgramFileFromSource(const std::string& programFileName);
            /**
             * load program from precompiled binary
             * @param kernelFileName
             * @return 
             */
            bool loadProgramFromBinary(const std::string& programFileName);
            /**
             * saves program binary loaded from source
             */
            char* saveProgramBinary(const std::string& programFileName);
            /**
             * return class names used in config
             */
            virtual std::string getClassName() = 0;
        public:
            OpenClBase();
            virtual ~OpenClBase();                        
            /**
             * init global openCL variables
             */
            virtual void initVars();
            /**
             * init openCL variables necessary for one image processing
             */
            virtual void initWorkVars() = 0;
            /**
             * clean up global variables
             */
            virtual void cleanUp();
            /**
             * clean up variables used for single image processing
             */
            virtual void cleanWorkPart() = 0;
            /**
             * return is called init method;
             * @return 
             */
            bool hasInitialized();
            /**
             * init variables for OpenclTools class instances
             * @param platformID
             * @param deviceID
             * @param listOnly
             */
            virtual void init(uint platformID, uint deviceID, bool listOnly) throw (SDException&);
        };
        
    }
}

#endif

#endif
