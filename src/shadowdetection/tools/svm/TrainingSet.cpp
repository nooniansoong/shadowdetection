#include "TrainingSet.h"
#include <fstream>
#include <iostream>
#include "shadowdetection/util/raii/RAIIS.h"
#include "typedefs.h"

#include "opencv2/ocl/ocl.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#define SPACES_COUNT 2
#define HSV_PARAMETERS 6
#define HLS_PARAMETERS 6

namespace shadowdetection {
    namespace tools {
        namespace svm {

            using namespace std;
            using namespace shadowdetection::util::raii;
            using namespace cv;

            void TrainingSet::readFile()throw (SDException&) {
                fstream file;
                file.open(filePath.c_str(), ifstream::in);
                FileRaii fRaii(&file);
                if (file.is_open()) {
                    string line;
                    while (getline(file, line)) {
                        vector<string> tokens = split(line, '\t');
                        if (tokens.size() >= 2) {
                            KeyVal<string> pair(tokens[0], tokens[1]);
                            images.push_back(pair);
                        }
                    }
                } else {
                    SDException exc(SHADOW_READ_UNABLE, "Learning: Read csv file");
                    throw exc;
                }
            }

            string getStr(float* arr, int dimension) {
                stringstream stream;
                for (int i = 0; i < dimension; i++) {
                    if (i == 0) {
                        stream << arr[i];
                    } else {
                        stream << " " << i << ":" << arr[i];
                    }
                }
                string retStr = stream.str();
                return retStr;
            }

            void TrainingSet::processImages(string output) throw (SDException&) {
                fstream file;
                file.open(output.c_str(), fstream::out | fstream::trunc);
                FileRaii fRaii(&file);
                if (file.is_open()) {
                    size_t size = images.size();
                    bool first = true;
                    for (int i = 0; i < size; i++) {
                        KeyVal<string> pair = images[i];
                        cout << "processing: " << pair.getKey() << endl;
                        int dimension = 0;
                        int pixelNum = 0;
                        float** processed = processImage(pair.getKey(), pair.getVal(), dimension, pixelNum);
                        if (processed != 0) {
                            //to have same number of 1s and 0s
                            bool write0 = true;
                            for (int j = 0; j < pixelNum; j++) {                                
                                bool succ = false;
                                if (write0){
                                    if (processed[j][0] == 0.f)
                                        succ = true;
                                }
                                else{
                                    if (processed[j][0] != 0.f)
                                        succ = true;
                                }
                                if (succ){
                                    string str = getStr(processed[j], dimension);
                                    string tmp = str.substr(0, 2);
                                    if (tmp.compare("1 ") != 0 && tmp.compare("0 ") != 0){
                                        cout << "Error create train set, label value: " << tmp << endl;
                                    }
                                    
                                    if (first == false) {
                                        file << endl;
                                    }
                                    file << str;
                                    first = false;
                                    write0 = !write0;
                                }
                                delete[] processed[j];
                            }
                            delete[] processed;
                        }
                    }
                } else {
                    SDException exc(SHADOW_WRITE_UNABLE, "Learning: process images");
                    throw exc;
                }
            }

            float clamp(float val, float min, float max) {
                if (val > max)
                    return max;
                if (val < min)
                    return min;
                return val;
            }

            float* processHSV(uchar H, uchar S, uchar V, int& size) {
                size = HSV_PARAMETERS;
                float* retArr = new float[size];
                //180 is max in opencv for H
                retArr[0] = (float) H / 180.f;
                retArr[0] = clamp(retArr[0], 0.f, 1.f);
                retArr[1] = (float) S / 255.f;
                retArr[1] = clamp(retArr[1], 0.f, 1.f);
                retArr[2] = (float) V / 255.f;
                retArr[2] = clamp(retArr[2], 0.f, 1.f);
                retArr[3] = (float) H / (float) (S + 1);
                retArr[3] /= 180.f;
                retArr[3] = clamp(retArr[3], 0.f, 1.f);
                retArr[4] = (float) H / (float) (V + 1);
                retArr[4] /= 180.f;
                retArr[4] = clamp(retArr[4], 0.f, 1.f);
                retArr[5] = (float) S / (float) (V + 1);
                retArr[5] /= 255.f;
                retArr[5] = clamp(retArr[5], 0.f, 1.f);
                return retArr;
            }

            float* processHLS(uchar H, uchar L, uchar S, int& size) {
                size = HLS_PARAMETERS;
                float* retArr = new float[size];
                //180 is max in opencv for H
                retArr[0] = (float) H / 180.f;
                retArr[0] = clamp(retArr[0], 0.f, 1.f);
                retArr[1] = (float) L / 255.f;
                retArr[1] = clamp(retArr[1], 0.f, 1.f);
                retArr[2] = (float) S / 255.f;
                retArr[2] = clamp(retArr[2], 0.f, 1.f);
                retArr[3] = (float) H / (float) (L + 1);
                retArr[3] /= 180.f;
                retArr[3] = clamp(retArr[3], 0.f, 1.f);
                retArr[4] = (float) H / (float) (S + 1);
                retArr[4] /= 180.f;
                retArr[4] = clamp(retArr[4], 0.f, 1.f);
                retArr[5] = (float) L / (float) (S + 1);
                retArr[5] /= 255.f;
                retArr[5] = clamp(retArr[5], 0.f, 1.f);
                return retArr;
            }

            float* merge(float label, float** arrs, int arrsLen, int* arrSize, int& retSize) {
                retSize = 1;
                for (int i = 0; i < arrsLen; i++) {
                    retSize += arrSize[i];
                }
                float* retArr = new float[retSize];
                int counter = 0;
                retArr[counter] = label;
                for (int i = 0; i < arrsLen; i++) {
                    int size = arrSize[i];
                    float* arr = arrs[i];
                    for (int j = 0; j < size; j++) {
                        retArr[++counter] = arr[j];
                    }
                }
                return retArr;
            }

            float getLabel(uchar val) {
                if (val == 0)
                    return 0.f;
                else
                    return 1.f;
            }

            float** TrainingSet::processImage(std::string orImage, std::string maskImg, int& rowDimesion, int& pixelNum) {
                Mat originalImage = cv::imread(orImage, CV_LOAD_IMAGE_COLOR);
                Mat maskImage = cv::imread(maskImg, CV_LOAD_IMAGE_GRAYSCALE);

                if (originalImage.data == 0 || maskImage.data == 0) {

                    return 0;
                }

                int d1 = originalImage.depth();
                int d2 = maskImage.depth();
                CV_8U;
                size_t maskStep = maskImage.step;
                int maskChan = maskImage.channels();

                if (originalImage.size().width == maskImage.size().width ||
                        originalImage.size().height == maskImage.size().height) {
                    int width = originalImage.size().width;
                    int height = originalImage.size().height;

                    float** retVec = new float*[width * height];

                    Mat hsv;
                    cvtColor(originalImage, hsv, CV_BGR2HSV);
                    size_t stepHsv = hsv.step;
                    int chanHSV = hsv.channels();

                    Mat hls;
                    cvtColor(originalImage, hls, CV_BGR2HLS);
                    size_t stepHls = hls.step;
                    int chanHLS = hls.channels();


                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            uchar hHSV = hsv.data[i * stepHsv + j * chanHSV + 0];
                            uchar sHSV = hsv.data[i * stepHsv + j * chanHSV + 1];
                            uchar vHSV = hsv.data[i * stepHsv + j * chanHSV + 2];

                            uchar hHLS = hls.data[i * stepHls + j * chanHLS + 0];
                            uchar lHLS = hls.data[i * stepHls + j * chanHLS + 1];
                            uchar sHLS = hls.data[i * stepHls + j * chanHLS + 1];

                            float* procs[SPACES_COUNT];
                            int size[SPACES_COUNT];
                            procs[0] = processHSV(hHSV, sHSV, vHSV, size[0]);
                            procs[1] = processHLS(hHLS, lHLS, sHLS, size[1]);

                            float label = getLabel(maskImage.data[i * maskStep + j * maskChan]);
                            int mergedSize = 0;
                            float* merged = merge(label, procs, 2, size, mergedSize);

                            for (int k = 0; k < SPACES_COUNT; k++)
                                delete[] procs[k];
                            retVec[i * width + j] = merged;
                            if (i == 0 && j == 0) {
                                rowDimesion = mergedSize;
                                pixelNum = width * height;
                            }
                        }
                    }
                    return retVec;
                } else {
                    return 0;
                }
            }

            TrainingSet::TrainingSet() {

            }

            TrainingSet::~TrainingSet() {
                clear();
            }

            TrainingSet::TrainingSet(std::string filePath) {
                setFilePath(filePath);
            }

            void TrainingSet::setFilePath(std::string filePath) {
                this->filePath = filePath;
            }

            void TrainingSet::process(string output) throw (SDException&) {
                readFile();
                processImages(output);
            }

            void TrainingSet::clear() {
                images.clear();
            }

        }
    }
}
