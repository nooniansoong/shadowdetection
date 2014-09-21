#ifndef __OPENCL_BASE_H__
#define __OPENCL_BASE_H__

#ifdef _OPENCL

#include <CL/cl.h>
#include "typedefs.h"
#include <vector>

namespace core{
    namespace opencl{
        
        /**
         * helper struct, translation of libsvm structure
         */
        struct cl_svm_node{
            cl_int index;
            cl_double value;
        };
        
        /**
         * helper struct, translation of libsvm structure with float instead of double value
         */
        struct cl_svm_node_float{
            cl_int index;
            cl_float value;
        };
        
        /**
         * Base class for opencl tools. All "OpenCL" classes should be derived from this one, and then specified in configuration file
         */
        class OpenClBase{
        private:
            std::string dirToOpenclprogramFiles;
            
            /**
             * compiles program file
             * @param programFileName
             * input file to compile
             * @return 
             * path to compiled file
             */
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
            std::vector<std::string> getKernelNamesForClass();
            /**
             * return list of source file for class specified in config
             * @return 
             */
            std::vector<std::string> getProgramFilesForClass();
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
             * load OpenCL program from source
             * @param kernelFileName
             */
            void loadProgramFileFromSource(const std::string& programFileName);
            /**
             * load OpenCL program from precompiled binary
             * @param kernelFileName
             * @return 
             */
            bool loadProgramFromBinary(const std::string& programFileName);
            /**
             * saves compiled OpenCL program binary loaded from source
             */
            char* saveProgramBinary(const std::string& programFileName);
            /**
             * return class names used in config
             */
            virtual std::string getClassName() = 0;
        public:
            /**
             * constructor. Calls initVars()
             */
            OpenClBase();
            /**
             * destructor call cleanUp()
             */
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
            void init(uint platformID, uint deviceID, bool listOnly) throw (SDException&);
        };
        
    }
}

#endif

#endif
