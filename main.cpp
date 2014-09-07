/* 
 * File:   main.cpp
 * Author: marko
 *
 * Created on February 14, 2014, 7:36 PM
 */

#include <iostream>
#include <memory>
#include "core/util/Config.h"
#include "core/tools/svm/TrainingSet.h"
#include "core/tools/svm/libsvmopenmp/svm-train.h"
#include "core/util/Matrix.h"
#include "core/opencl/libsvm/OpenCLToolsTrain.h"
#include "shadowdetection/opencl/OpenCLTools.h"
#include "core/process/IProcessor.h"
#include "core/util/rtti/ObjectFactory.h"
#include "core/util/rtti/ObjectFactory.h"

using namespace std;
using namespace core::util;
using namespace core::tools::svm;
#ifdef _OPENCL
using namespace core::opencl::libsvm;
using namespace shadowdetection::opencl;
#endif
using namespace core::tools::svm::libsvmopenmp;
using namespace core::process;
using namespace core::util::RTTI;

/**
 * global function for process single image
 * @param input
 * @param out
 */

int main(int argc, char **argv) {
    cout << "MAIN" << endl;    
    if (argc == 1){
        cout << "Call with -help for help" << endl;
#ifdef _OPENCL
        cout << "Call with -list for list of openCL capable platforms" << endl;
#endif
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "-help") == 0){
        cout << "Please visit: http://code.google.com/p/shadowdetection/wiki/ShadowDetection section Usage" << endl;
        return 0;
    }

#ifdef _OPENCL
    if (argc == 2 && strcmp(argv[1], "-list") == 0){
        try{
            OpenclTools::getInstancePtr()->init(0, 0, true);
            OpenclTools::getInstancePtr()->cleanUp();
        } catch (SDException& exception) {
            cout << exception.handleException() << endl;
            exit(1);
        }
        return 0;
    }
#endif         
    
    if (argc >= 2 && strcmp(argv[1], "-makeset") == 0){
        if (argc < 4){
            cout << "makeset needs more parameters: input csv file, output file" << endl;
#ifdef _OPENCL
            OpenclTools::getInstancePtr()->cleanUp();
#endif            
            return 0;
        }
        try{
            TrainingSet ts(argv[2]);
            bool distribute = true;
            string distributeStr = Config::getInstancePtr()->getPropertyValue("process.Training.distribute0and1");
            if (distributeStr.compare("false") == 0){
                distribute = false;
            }
            ts.process(argv[3], !distribute);
        }
        catch (SDException& exc){
            cout << exc.handleException() << endl;
#ifdef _OPENCL
            OpenclTools::getInstancePtr()->cleanUp();
#endif            
            exit(1);
        }
#ifdef _OPENCL
        OpenclTools::getInstancePtr()->cleanUp();
#endif
        return 0;
    }
        
    if (argc >= 2 && strcmp(argv[1], "-training") == 0) {
        if (argc < 4){
            cout << "training needs more parameters: input data file, output file" << endl;
            return 0;
        }
        try{
            Config* conf = Config::getInstancePtr();
#ifdef _OPENCL
            int platformId = 0;
            int deviceId = 0;
            string platformStr = conf->getPropertyValue("general.openCL.platformid");
            string deviceStr = conf->getPropertyValue("general.openCL.deviceid");
            int tmp = atoi(platformStr.c_str());
            if (tmp != 0)
                platformId = tmp;
            tmp = atoi(deviceStr.c_str());
            if (tmp != 0)
                deviceId = tmp;
            OpenCLToolsTrain::getInstancePtr()->init(platformId, deviceId, false);
#endif
            int val = train(argv[2], argv[3]);
            cout << val << endl;
        }
        catch (SDException& e){
            cout << e.handleException() << endl;
            exit(1);
        }
#ifdef _OPENCL
        OpenclTools::getInstancePtr()->cleanUp();
        OpenclTools::destroy();        
#endif
        Config::destroy();
        return 0;
    }
    
    //TODO instance ShadowDetection processor
    {
        UNIQUE_PTR(IProcessor) ip(ObjectFactory::getInstancePtr()->createInstance<IProcessor>("shadowdetection::process::ShadowDetectionProcessor"));
        ip->init();
        ip->process(argc, argv);
    }
#ifdef _DEBUG    
    string unallocated = MemTracker::getUnfreed();
    cout << unallocated << endl;
#endif
    return 0;
}

