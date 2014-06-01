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

void handleError(int exception){
    cout << "Error: " << exception << endl;
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
    } catch (int exception) {
        throw exception;
    }
    if (processedImage != 0) {
        imwrite(out, *processedImage);
        delete processedImage;
    }
}
#endif

IplImage* image;
void processSingle(const char* input, const char* out) throw (int) {        
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
        catch (int exception) {
            throw exception;
        }        
    } else {
        throw (int)SHADOW_READ_UNABLE;
    }    
}

int main(int argc, char **argv) {
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
    catch (int exception){
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
            catch (int exception){
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
            catch (int exception){
                handleError(exception);
                exit(1);
            }
            for (int i = 0; i < tp.size(); i++){
                string in = tp.get(i).getKey();
                string out = tp.get(i).getVal();
                try{
                    processSingle(in.c_str(), out.c_str());
                }
                catch (int exception){
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

