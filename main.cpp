/* 
 * File:   main.cpp
 * Author: marko
 *
 * Created on February 14, 2014, 7:36 PM
 */

#include <iostream>
#include "shadowdetection/opencl/OpenCLTools.h"
#include "shadowdetection/opencv/OpenCVTools.h"
#include "shadowdetection/util/Config.h"
#include "shadowdetection/util/TabParser.h"
#include "shadowdetection/util/raii/RAIIS.h"

using namespace std;
#ifdef _OPENCL
using namespace shadowdetection::opencl;
#endif
using namespace shadowdetection::opencv;
using namespace shadowdetection::util;
using namespace cv;
using namespace shadowdetection::util::raii;

void handleError(const SDException& exception){
    const char* err = exception.what();
    cout << "Error: " << err << endl;
}

#ifndef _OPENCL
void processSingleCPU(const char* out, IplImage* image) {
    IplImage* processedImage = 0;
    int height, width, channels;
    unsigned int* hsi1 = OpenCvTools::convertImagetoHSI(image, height, width, channels, &OpenCvTools::RGBtoHSI_1);
    unsigned char* ratios1 = OpenCvTools::simpleTsai(hsi1, height, width, channels);
    IplImage* ratiosImage1 = OpenCvTools::get8bitImage(ratios1, height, width);
    IplImage* binarized1 = OpenCvTools::binarize(ratiosImage1);
    unsigned int* hsi2 = OpenCvTools::convertImagetoHSI(image, height, width, channels, &OpenCvTools::RGBtoHSI_2);
    unsigned char* ratios2 = OpenCvTools::simpleTsai(hsi2, height, width, channels);
    IplImage* ratiosImage2 = OpenCvTools::get8bitImage(ratios2, height, width);
    IplImage* binarized2 = OpenCvTools::binarize(ratiosImage2);

    processedImage = OpenCvTools::joinTwo(binarized1, binarized2);

    delete[] hsi1;
    delete[] ratios1;
    delete[] hsi2;
    delete[] ratios2;

    cvReleaseImage(&binarized1);
    cvReleaseImage(&ratiosImage1);
    cvReleaseImage(&binarized2);
    cvReleaseImage(&ratiosImage2);    
    cvSaveImage(out, processedImage);
    cvReleaseImage(&processedImage);
}
#endif

#ifdef _OPENCL
OpenclTools oclt;
void processSingleGPU(const char* out, IplImage* image) {                
    unsigned char* buffer = OpenCvTools::convertImageToByteArray(image);
    Mat* processedImage = 0;
    try {
        processedImage = oclt.processRGBImage(buffer, image->width, image->height, image->nChannels);        
    } catch (SDException& exception) {
        throw exception;
    }
    if (processedImage != 0) {
        imwrite(out, *processedImage);
        delete processedImage;
    }
}
#endif

IplImage* image;
void processSingle(const char* input, const char* out) throw (SDException&) {        
    image = cvLoadImage(input);
    ImageRaii rai(image);    
    if (image != 0) {
        try {
#ifdef _OPENCL            
            processSingleGPU(out, image);
#else            
            processSingleCPU(out, image);
#endif
        }            
        catch (SDException& exception) {
            throw exception;
        }        
    } else {
        string msg = "Process single image file: ";
        msg += input;
        SDException exc(SHADOW_READ_UNABLE, msg);
        throw exc;
    }    
}

int main(int argc, char **argv) {
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
            oclt.init(0, 0);
            oclt.cleanUp();
        } catch (SDException& exception) {
            handleError(exception);
            exit(1);
        }
        return 0;
    }
#endif
    Config* conf = Config::getInstancePtr();
    string useBatch = conf->getPropertyValue("process.UseBatch");
#ifdef _OPENCL    
    try{
        int platformId = 0;
        int deviceId = 0;
        string platformStr = conf->getPropertyValue("settings.openCL.platformid");
        string deviceStr = conf->getPropertyValue("settings.openCL.platformid");
        int tmp = atoi(platformStr.c_str());
        if (tmp != 0)
            platformId = tmp;
        tmp = atoi(deviceStr.c_str());
        if (tmp != 0)
            deviceId = tmp;
        oclt.init(platformId, deviceId);
        OpenCvTools::initOpenCL(platformId, deviceId);        
    }
    catch (SDException& exception){
        handleError(exception);
        exit(1);
    }
#endif
    if (useBatch.compare("false") == 0){
        if (argc > 2) {
            char* path = argv[1];
            char* savePath = argv[2];
            try{
                processSingle(path, savePath);
            }
            catch (SDException& exception){
                handleError(exception);
                exit(1);
            }
        }
    }
    else{
        if (argc > 1){
            char* path = argv[1];
            TabParser tp;
            try{
                tp.init(path);
            }
            catch (SDException& exception){
                handleError(exception);
                exit(1);
            }
            for (int i = 0; i < tp.size(); i++){
                string in = tp.get(i).getKey();
                string out = tp.get(i).getVal();
                try{
                    processSingle(in.c_str(), out.c_str());
                }
                catch (SDException& exception){
                    handleError(exception);
                }
#ifdef _OPENCL
                oclt.cleanWorkPart();
#endif
            }
            
        }
    }
#ifdef _OPENCL
    oclt.cleanUp();
#endif
    return 0;
}

