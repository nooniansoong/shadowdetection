#include "ShadowDetectionProcessor.h"
#include <string>
#include "shadowdetection/opencl/OpenCLTools.h"
#include "core/opencl/libsvm/OpenCLToolsPredict.h"
#include "shadowdetection/opencl/OpenCLImageParameters.h"
#include "core/util/Config.h"
#include "core/opencv/OpenCV2Tools.h"
#include "core/opencv/OpenCVTools.h"
#if defined _OPENMP_MY
#include <omp.h>
#endif
#include "shadowdetection/tools/image/ResultFixer.h"
#include "core/util/raii/RAIIS.h"
#include "core/tools/image/IImageParameters.h"
#include "core/util/rtti/ObjectFactory.h"
#include "core/util/predicition/IPrediction.h"
#include "core/util/TabParser.h"

namespace shadowdetection {
    namespace process {

        REGISTER_CLASS(ShadowDetectionProcessor, shadowdetection::process)

#ifdef _OPENCL
        using namespace shadowdetection::opencl;
        using namespace core::opencl::libsvm;
#endif
        using namespace std;
        using namespace core::util;
        using namespace core::util::raii;
        using namespace core::util::RTTI;
        using namespace core::util::prediction;
        using namespace core::tools::image;
        using namespace core::opencv;
        using namespace core::opencv2;
        using namespace cv;
        using namespace shadowdetection::tools::image;

        void initOpenCL() {
#ifdef _OPENCL
            try {
                int platformId = 0;
                int deviceId = 0;
                Config* conf = Config::getInstancePtr();
                string platformStr = conf->getPropertyValue("general.openCL.platformid");
                string deviceStr = conf->getPropertyValue("general.openCL.deviceid");
                int tmp = atoi(platformStr.c_str());
                if (tmp != 0)
                    platformId = tmp;
                tmp = atoi(deviceStr.c_str());
                if (tmp != 0)
                    deviceId = tmp;
                OpenclTools::getInstancePtr()->init(platformId, deviceId, false);
                OpenCV2Tools::initOpenCL(platformId, deviceId);
                OpenCLToolsPredict::getInstancePtr()->init(platformId, deviceId, false);
                OpenCLImageParameters::getInstancePtr()->init(platformId, deviceId, false);
            }      
            catch (SDException& exception) {
                cout << exception.handleException() << endl;
                exit(1);
            }
#endif
        }

        void initOpenMP() {
#if defined _OPENMP_MY
            omp_set_dynamic(0);
            int numThreads = 4;
            string tnStr = Config::getInstancePtr()->getPropertyValue("general.openMP.threadNum");
            int tmp = atoi(tnStr.c_str());
            if (tmp != 0)
                numThreads = tmp;
            omp_set_num_threads(numThreads);
#endif
        }

#ifdef _OPENCL

        /**
         * process single image using openCL
         * @param out
         * @param imageNew
         */
        void processSingleOpenCL(const char* out, const Mat& image) {
            UNIQUE_PTR(Mat) hlsPtr(OpenCV2Tools::convertToHLS(&image));
            if (hlsPtr.get() == 0) {
                return;
            }
            OpenclTools* oclt = OpenclTools::getInstancePtr();
            uchar* buffer = OpenCV2Tools::convertImageToByteArray(&image, true);
            VectorRaii<uchar> bufferRaii(buffer);
            UNIQUE_PTR(Mat) processedImagePtr;
            bool usePrediction = false;
            string usePredStr = Config::getInstancePtr()->getPropertyValue("general.Prediction.usePrediction");
            if (usePredStr.compare("true") == 0)
                usePrediction = true;
            if (usePrediction == false) {
                try {
                    processedImagePtr = UNIQUE_PTR(Mat)(oclt->processRGBImage(buffer, image.size().width,
                            image.size().height, image.channels()));
                } catch (SDException& exception) {
                    throw exception;
                }
            } else {
                UNIQUE_PTR(Mat) piPtr(oclt->processRGBImage(buffer, image.size().width, image.size().height, image.channels()));
                if (piPtr.get()) {
                    int pixCount;
                    int parameterCount;
                    UNIQUE_PTR(IImageParameteres) ipPtr(ObjectFactory::getInstancePtr()->createImageParameters());

                    UNIQUE_PTR(Mat) hsvPtr(OpenCV2Tools::convertToHSV(&image));
                    if (hsvPtr.get() == 0) {
                        return;
                    }

                    UNIQUE_PTR(Matrix<float>) parametersPtr(ipPtr->getImageParameters(image, *hsvPtr, *hlsPtr,
                            parameterCount, pixCount));
                    if (parametersPtr.get() != 0) {
                        IPrediction* predictor = ObjectFactory::getInstancePtr()->createPredictor();
                        if (predictor->hasLoadedModel() == false) {
                            predictor->loadModel();
                        }
                        uchar* predicted = predictor->predict(parametersPtr.get(), pixCount, parameterCount);
                        if (predicted) {
                            VectorRaii<uchar> vraiiPred(predicted);
                            //FileSaver<uchar>::saveToFile("myPredictedOCL.out", predicted, pixCount);
                            for (int i = 0; i < pixCount; i++)
                                predicted[i] *= 255;
                            UNIQUE_PTR(Mat) predictedImagePtr(OpenCV2Tools::get8bitImage(predicted,
                                    image.size().height, image.size().width));
                            processedImagePtr = UNIQUE_PTR(Mat)(OpenCV2Tools::joinTwoOcl(*piPtr, *predictedImagePtr));
                        } else {
                            SDException e(SHADOW_CANT_PREDICT, "processSingleOpenCL");
                            throw e;
                        }
                    } else {
                        SDException e(SHADOW_CANT_GET_PARAMETERS, "processSingleOpenCL");
                        throw e;
                    }
                }
            }
            if (processedImagePtr.get() != 0) {
                ResultFixer rf;
                rf.applyThreshholds(*processedImagePtr, image, *hlsPtr);
                imwrite(out, *processedImagePtr);
            }
        }
#else
        /**
        * process single image on CPU not using openCL
        * @param out
        * @param image
        */
       void processSingleCPU(const char* out, IplImage* image) {    
           Mat imageMat(image);
           UNIQUE_PTR(Mat) hls(OpenCV2Tools::convertToHLS(&imageMat));
           if (hls == 0){
               return;
           }    

           IplImage* processedImage = 0;
           int height, width, channels;
           uint* hsi1 = OpenCvTools::convertImagetoHSI(image, height, width, channels, &OpenCvTools::RGBtoHSI_1);
           VectorRaii<uint> vraiiHsi1(hsi1);
           uchar* ratios1 = OpenCvTools::simpleTsai(hsi1, height, width, channels);
           VectorRaii<uchar> vraiiR1(ratios1);
           IplImage* ratiosImage1 = OpenCvTools::get8bitImage(ratios1, height, width);
           ImageRaii iariiR1(ratiosImage1);
           IplImage* binarized1 = OpenCvTools::binarize(ratiosImage1);
           ImageRaii iraiiBin1(binarized1);
           uint* hsi2 = OpenCvTools::convertImagetoHSI(image, height, width, channels, &OpenCvTools::RGBtoHSI_2);
           VectorRaii<uint> vraiiHsi2(hsi2);
           uchar* ratios2 = OpenCvTools::simpleTsai(hsi2, height, width, channels);
           VectorRaii<uchar> vraiiR2(ratios2);
           IplImage* ratiosImage2 = OpenCvTools::get8bitImage(ratios2, height, width);
           ImageRaii iraiiR2(ratiosImage2);
           IplImage* binarized2 = OpenCvTools::binarize(ratiosImage2);
           ImageRaii iraiiBin2(binarized2);

           bool usePrediction = false;
           string usePredStr = Config::getInstancePtr()->getPropertyValue("general.Prediction.usePrediction");
           if (usePredStr.compare("true") == 0)
               usePrediction = true;
           if (usePrediction){
               IplImage* pi = OpenCvTools::joinTwo(binarized1, binarized2);
               ImageRaii iraii(pi);        
               int pixCount;
               int parameterCount;

               UNIQUE_PTR(Mat) hsv(OpenCV2Tools::convertToHSV(&imageMat));
               if (hsv == 0){
                   return;
               }        

               UNIQUE_PTR(IImageParameteres) ipPtr(ObjectFactory::getInstancePtr()->createImageParameters());        
               UNIQUE_PTR(Matrix<float>) parameters(ipPtr->getImageParameters(imageMat, *hsv, *hls, 
                                                                               parameterCount, pixCount));
               IPrediction* predictor = ObjectFactory::getInstancePtr()->createPredictor();
               if (predictor->hasLoadedModel() == false){            
                   predictor->loadModel();
               }
               uchar* predicted = predictor->predict(parameters.get(), pixCount, parameterCount);
               VectorRaii<uchar> vraii(predicted);        
               for (int i = 0; i < pixCount; i++)
                   predicted[i] *= 255;
               IplImage* predictedImage = OpenCvTools::get8bitImage(predicted, height, width);
               ImageRaii iraii2(predictedImage);        
               processedImage = OpenCvTools::joinTwo(pi, predictedImage);               
           }
           else{
               processedImage = OpenCvTools::joinTwo(binarized1, binarized2);
           }
           if (processedImage){
               ImageRaii iraii(processedImage);
               ResultFixer rf;
               Mat processedImageMat(processedImage);
               rf.applyThreshholds(processedImageMat, imageMat, *hls);
               cvSaveImage(out, processedImage);
           }
        }
#endif

        void cleanUp(){
#ifdef _OPENCL
        OpenclTools::getInstancePtr()->cleanUp();            
        OpenCLToolsPredict::getInstancePtr()->cleanUp();
        OpenCLImageParameters::getInstancePtr()->cleanUp();
        OpenclTools::destroy();            
        OpenCLToolsPredict::destroy();
        OpenCLImageParameters::destroy();            
#endif
        Config::destroy();
        }
        
        void cleanUpWork(){
#ifdef _OPENCL
            OpenclTools::getInstancePtr()->cleanWorkPart();
            OpenCLToolsPredict::getInstancePtr()->cleanWorkPart();
            OpenCLImageParameters::getInstancePtr()->cleanWorkPart();
#endif
        }
       
        void processSingle(const char* input, const char* out) throw (SDException&) {
            cout << "===========" << endl;
            cout << "Processing: " << input << endl;
            try {
#ifdef _OPENCL
                Mat imageNew = cv::imread(input);
                if (imageNew.data == 0) {
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
                } else {
                    string msg = "Process single image file: ";
                    msg += input;
                    SDException exc(SHADOW_READ_UNABLE, msg);
                    throw exc;
                }
#endif
            } catch (SDException& exception) {
                throw exception;
            }
        }

        ShadowDetectionProcessor::ShadowDetectionProcessor() : IProcessor() {

        }

        ShadowDetectionProcessor::~ShadowDetectionProcessor() {
            cleanUp();
        }

        void ShadowDetectionProcessor::init() throw (SDException&) {
            initOpenCL();
            initOpenMP();
        }
        
        void ShadowDetectionProcessor::process(int argc, char **argv) {
            Config* conf = Config::getInstancePtr();
            string useBatch = conf->getPropertyValue("general.UseBatch");
            if (useBatch.compare("false") == 0) {
                if (argc > 2) {
                    char* path = argv[1];
                    char* savePath = argv[2];
                    try {
                        processSingle(path, savePath);
                    } catch (SDException& exception) {
                        cout << exception.handleException() << endl;                        
                        return;
                    }
                } else {                    
                    cout << "need two arguments: input image path, output image path" << endl;
                    return;
                }
            } else {
                if (argc > 1) {
                    char* path = argv[1];
                    TabParser tp;
                    try {
                        tp.init(path);
                    } catch (SDException& exception) {
                        cout << exception.handleException() << endl;
                        exit(1);
                    }
                    for (uint i = 0; i < tp.size(); i++) {
                        string in = tp.get(i).getFirst();
                        string out = tp.get(i).getSecond();
                        try {
                            processSingle(in.c_str(), out.c_str());
                        } catch (SDException& exception) {
                            cout << exception.handleException() << endl;
                            cout << "Continue to process" << endl;
                        }
                        cleanUpWork();
                    }                    
                } else {
                    cout << "Needed parameter path to csv file" << endl;                    
                    return;
                }
            }
        }
    }
}
