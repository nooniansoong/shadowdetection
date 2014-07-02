/* 
 * File:   main.cpp
 * Author: marko
 *
 * Created on February 14, 2014, 7:36 PM
 */

#include <iostream>
#if !defined _MAC && !defined _OPENCL
#include <omp.h>
#endif
#include "shadowdetection/opencl/OpenCLTools.h"
#include "shadowdetection/opencv/OpenCV2Tools.h"
#include "shadowdetection/opencv/OpenCVTools.h"
#include "shadowdetection/util/Config.h"
#include "shadowdetection/util/TabParser.h"
#include "shadowdetection/util/raii/RAIIS.h"
#include "shadowdetection/tools/svm/TrainingSet.h"
#include "shadowdetection/tools/svm/libsvmopenmp/svm-train.h"
#include "shadowdetection/util/libsvm/SvmPredict.h"
#include "shadowdetection/util/image/ImageParameters.h"
#include "shadowdetection/util/FileSaver.h"
#include "shadowdetection/util/Matrix.h"

using namespace std;
#ifdef _OPENCL
using namespace shadowdetection::opencl;
#endif
using namespace shadowdetection::opencv;
using namespace shadowdetection::opencv2;
using namespace shadowdetection::util;
using namespace cv;
using namespace shadowdetection::util::raii;
using namespace shadowdetection::tools::svm;
using namespace shadowdetection::util::libsvm;
using namespace shadowdetection::util::image;

void initOpenMP(){
#if !defined _MAC && !defined _OPENCL
    omp_set_dynamic(0);
    int numThreads = 4;                    
    string tnStr = Config::getInstancePtr()->getPropertyValue("settings.openMP.threadNum");
    int tmp = atoi(tnStr.c_str());
    if (tmp != 0)
        numThreads = tmp;
    omp_set_num_threads(numThreads);
#endif
}

#ifdef _OPENCL
void initOpenCL(){
    try{
        int platformId = 0;
        int deviceId = 0;
        Config* conf = Config::getInstancePtr();
        string platformStr = conf->getPropertyValue("settings.openCL.platformid");
        string deviceStr = conf->getPropertyValue("settings.openCL.deviceid");
        int tmp = atoi(platformStr.c_str());
        if (tmp != 0)
            platformId = tmp;
        tmp = atoi(deviceStr.c_str());
        if (tmp != 0)
            deviceId = tmp;
        OpenclTools::getInstancePtr()->init(platformId, deviceId, false);
        OpenCV2Tools::initOpenCL(platformId, deviceId);        
    }
    catch (SDException& exception){
        handleException(exception);
        exit(1);
    }
}
#endif

void handleException(const SDException& exception){
    const char* err = exception.what();
    cout << "Error: " << err << endl;
}

#ifndef _OPENCL
/**
 * process single image on CPU not using openCL
 * @param out
 * @param image
 */
void processSingleCPU(const char* out, IplImage* image) {    
    IplImage* processedImage = 0;
    int height, width, channels;
    unsigned int* hsi1 = OpenCvTools::convertImagetoHSI(image, height, width, channels, &OpenCvTools::RGBtoHSI_1);
    VectorRaii vraiiHsi1(hsi1);
    unsigned char* ratios1 = OpenCvTools::simpleTsai(hsi1, height, width, channels);
    VectorRaii vraiiR1(ratios1);
    IplImage* ratiosImage1 = OpenCvTools::get8bitImage(ratios1, height, width);
    ImageRaii iariiR1(ratiosImage1);
    IplImage* binarized1 = OpenCvTools::binarize(ratiosImage1);
    ImageRaii iraiiBin1(binarized1);
    unsigned int* hsi2 = OpenCvTools::convertImagetoHSI(image, height, width, channels, &OpenCvTools::RGBtoHSI_2);
    VectorRaii vraiiHsi2(hsi2);
    unsigned char* ratios2 = OpenCvTools::simpleTsai(hsi2, height, width, channels);
    VectorRaii vraiiR2(ratios2);
    IplImage* ratiosImage2 = OpenCvTools::get8bitImage(ratios2, height, width);
    ImageRaii iraiiR2(ratiosImage2);
    IplImage* binarized2 = OpenCvTools::binarize(ratiosImage2);
    ImageRaii iraiiBin2(binarized2);
    
    bool usePrediction = false;
    string usePredStr = Config::getInstancePtr()->getPropertyValue("process.Prediction.usePrediction");
    if (usePredStr.compare("true") == 0)
        usePrediction = true;
    if (usePrediction){
        IplImage* pi = OpenCvTools::joinTwo(binarized1, binarized2);
        ImageRaii iraii(pi);
        Mat* imageMat = new Mat(image);
        int pixCount;
        int parameterCount;
        Matrix<float>* parameters = ImageParameters::getImageParameters(*imageMat, parameterCount, pixCount);
        if (SvmPredict::getInstancePtr()->hasLoadedModel() == false){
            string modelFile = Config::getInstancePtr()->getPropertyValue("process.Prediction.modelFile");
            SvmPredict::getInstancePtr()->loadModel(modelFile);
        }
        uchar* predicted = SvmPredict::getInstancePtr()->predict(parameters, pixCount, parameterCount);
        VectorRaii vraii(predicted);
        //FileSaver<uchar>::saveToFile("myPredictedS1.out", predicted, pixCount);
        for (int i = 0; i < pixCount; i++)
            predicted[i] *= 255;
        IplImage* predictedImage = OpenCvTools::get8bitImage(predicted, height, width);
        ImageRaii iraii2(predictedImage);
        //cvSaveImage("predicted.jpg", predictedImage);
        processedImage = OpenCvTools::joinTwo(pi, predictedImage);               
    }
    else{
        processedImage = OpenCvTools::joinTwo(binarized1, binarized2);
    }
    ImageRaii iraii(processedImage);                
    cvSaveImage(out, processedImage);    
}
#endif

#ifdef _OPENCL
/**
 * process single image using openCL
 * @param out
 * @param imageNew
 */
void processSingleOpenCL(const char* out, const Mat& image) {                
    OpenclTools* oclt = OpenclTools::getInstancePtr();
    unsigned char* buffer = OpenCV2Tools::convertImageToByteArray(&image, true);    
    Mat* processedImage = 0;
    bool usePrediction = false;
    string usePredStr = Config::getInstancePtr()->getPropertyValue("process.Prediction.usePrediction");
    if (usePredStr.compare("true") == 0)
        usePrediction = true;
    if (usePrediction == false){
        try {
            processedImage = oclt->processRGBImage(buffer, image.size().width, image.size().height, image.channels());
        } catch (SDException& exception) {
            throw exception;
        }
    }
    else{
        Mat* pi = oclt->processRGBImage(buffer, image.size().width, image.size().height, image.channels());
        if (pi){
            ImageNewRaii imraiiPi(pi);
            int pixCount;
            int parameterCount;
            Matrix<float>* parameters = ImageParameters::getImageParameters(image, parameterCount, pixCount);
            if (parameters){
                PointerRaii< Matrix<float> > paramRaii(parameters);
                if (SvmPredict::getInstancePtr()->hasLoadedModel() == false){
                    string modelFile = Config::getInstancePtr()->getPropertyValue("process.Prediction.modelFile");
                    SvmPredict::getInstancePtr()->loadModel(modelFile);
                }
                uchar* predicted = SvmPredict::getInstancePtr()->predict(parameters, pixCount, parameterCount);
                if (predicted){
                    VectorRaii vraiiPred(predicted);
                    //FileSaver<uchar>::saveToFile("myPredictedOCL.out", predicted, pixCount);
                    for (int i = 0; i < pixCount; i++)
                        predicted[i] *= 255;
                    Mat* predictedImage = OpenCV2Tools::get8bitImage(predicted, image.size().height, image.size().width);
                    ImageNewRaii prdeRaii(predictedImage);
                    //imwrite("myPredictedOCL.jpg", *predictedImage);
                    processedImage = OpenCV2Tools::joinTwoOcl(*pi, *predictedImage);
                }
                else{
                    SDException e(SHADOW_CANT_PREDICT, "processSingleOpenCL");
                    throw e;
                }
            }
            else{
                SDException e(SHADOW_CANT_GET_PARAMETERS, "processSingleOpenCL");
                throw e;
            }
        }        
    }
    if (processedImage != 0) {
        imwrite(out, *processedImage);
        delete processedImage;
    }    
}
#endif

/**
 * global function for process single image
 * @param input
 * @param out
 */
void processSingle(const char* input, const char* out) throw (SDException&) {        
    cout << "===========" << endl;
    cout << "Processing: " << input << endl;            
    try {
#ifdef _OPENCL
        Mat imageNew = cv::imread(input);        
        if (imageNew.data == 0){
            string msg = "Process single image file: ";
            msg += input;
            SDException exc(SHADOW_READ_UNABLE, msg);
            throw exc;
        }
        processSingleOpenCL(out, imageNew);
#else
        IplImage* image = cvLoadImage(input);
        if (image != 0) {
            ImageRaii rai(image);
            processSingleCPU(out, image);
        }
        else {
            string msg = "Process single image file: ";
            msg += input;
            SDException exc(SHADOW_READ_UNABLE, msg);
            throw exc;
        } 
#endif
    }            
    catch (SDException& exception) {
        throw exception;
    }           
}

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
            handleException(exception);
            exit(1);
        }
        return 0;
    }
#endif
    
    initOpenMP();
#ifdef _OPENCL
    initOpenCL();
#endif
    
    if (argc >= 2 && strcmp(argv[1], "-makeset") == 0){
        if (argc < 4){
            cout << "makeset needs more parameters: input csv file, output file" << endl;
            return 0;
        }
        try{
            TrainingSet ts(argv[2]);
            bool distribute = true;
            string distributeStr = Config::getInstancePtr()->getPropertyValue("process.Training.distribute0and1");
            if (distributeStr.compare("false") == 0){
                distribute = false;
            }
            ts.process(argv[3], distribute);
        }
        catch (SDException& exc){
            handleException(exc);
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
            int val = shadowdetection::tools::svm::libsvmopenmp::train(argv[2], argv[3]);
            cout << val << endl;
        }
        catch (SDException& e){
            handleException(e);
            exit(1);
        }
#ifdef _OPENCL
        OpenclTools::getInstancePtr()->cleanUp();
#endif
        return 0;
    }
    
    Config* conf = Config::getInstancePtr();
    string useBatch = conf->getPropertyValue("process.UseBatch");
    if (useBatch.compare("false") == 0){
        if (argc > 2) {
            char* path = argv[1];
            char* savePath = argv[2];
            try{
                processSingle(path, savePath);
            }
            catch (SDException& exception){
                handleException(exception);
                exit(1);
            }
        }
        else{
            cout << "need two arguments: input image path, output image path" << endl;
            return 0;
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
                handleException(exception);
                exit(1);
            }
            for (int i = 0; i < tp.size(); i++){
                string in = tp.get(i).getKey();
                string out = tp.get(i).getVal();
                try{
                    processSingle(in.c_str(), out.c_str());
                }
                catch (SDException& exception){
                    handleException(exception);
                    cout << "Continue to process" << endl;
                }
#ifdef _OPENCL
                OpenclTools::getInstancePtr()->cleanWorkPart();
#endif
            }            
        }
        else{
            cout << "Needed parameter path to csv file" << endl;
            return 0;
        }
    }
#ifdef _OPENCL
    OpenclTools::getInstancePtr()->cleanUp();
#endif
    return 0;
}

