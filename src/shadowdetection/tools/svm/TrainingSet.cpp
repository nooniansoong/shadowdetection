#include "TrainingSet.h"
#include <fstream>
#include <iostream>
#include "shadowdetection/util/raii/RAIIS.h"
#include "typedefs.h"
#include "shadowdetection/util/image/ImageParameters.h"
#include "shadowdetection/opencv/OpenCV2Tools.h"

namespace shadowdetection {
    namespace tools {
        namespace svm {

            using namespace std;
            using namespace shadowdetection::util::raii;
            using namespace shadowdetection::util::image;
            using namespace cv;
            using namespace shadowdetection::opencv2;

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
                        cout << "Size: " << pixelNum;
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
                        else{
                            SDException exc(SHADOW_READ_UNABLE, "TrainingSet::processImages");
                            throw exc;
                        }
                    }
                } else {
                    SDException exc(SHADOW_WRITE_UNABLE, "TrainingSet::processImages");
                    throw exc;
                }
            }

            float** TrainingSet::processImage(std::string orImage, std::string maskImg, int& rowDimesion, int& pixelNum) {
                Mat originalImage = cv::imread(orImage, CV_LOAD_IMAGE_COLOR);
                Mat maskImage = cv::imread(maskImg, CV_LOAD_IMAGE_GRAYSCALE);

                if (originalImage.data == 0 || maskImage.data == 0) {

                    return 0;
                }
                float** retVec = ImageParameters::getImageParameters(originalImage, maskImage, rowDimesion, pixelNum);
                return retVec;
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
