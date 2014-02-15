/* 
 * File:   main.cpp
 * Author: marko
 *
 * Created on February 14, 2014, 7:36 PM
 */

#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <math.h>

using namespace std;

unsigned char maxF(unsigned char a, unsigned char b, unsigned char c) {
    unsigned char max = a;
    if (b > max)
        max = b;
    if (c > max)
        max = c;

    return max;
}

unsigned char minF(unsigned char a, unsigned char b, unsigned char c) {
    unsigned char min = a;
    if (b < min)
        min = b;
    if (c < min)
        min = c;

    return min;
}

float radToDegrees(float radians) {
    const float PI_F = 3.14159265358979f;
    float oneRad = 180. / PI_F;
    return radians * oneRad;
}

void RGBtoHSI_1(unsigned char r, unsigned char g, unsigned char b, unsigned int& h, unsigned char& s, unsigned char& i) {
    float min, max, delta;

    min = minF(r, g, b);
    max = maxF(r, g, b);
    i = (unsigned char) (((float) (r + g + b)) / 3.f);

    delta = max - min;
    float fS;
    float fH;

    float v1 = 0.5f * (2 * r - g - b);
    float v2 = 0.5f * sqrt(3) * (g - b);

    fH = atan2(v2, v1);
    fH = radToDegrees(fH);

    if (fH < -360.)
        fH = -360.;

    if (fH < 0.)
        fH += 360.;

    if (fH > 360.)
        fH = 360.;

    //fS =  (4 * b * b + 5 * g * g + 2 * r * r - 4 * g * b - 4 * b * r - 2 * g * r) / 6.f;//1 - 3 * ((float) min / (float) i);
    //fS = sqrt(fS);
    fS =  1 - 3 * ((float) min / (float) i);
    

    h = (unsigned int) fH;
    if (fS > 255.){
        cout << "JASOO " << fS << endl;
    }
    s = (unsigned char) (fS * 255.f);
}

void RGBtoHSI_2(unsigned char r, unsigned char g, unsigned char b, unsigned int& h, unsigned char& s, unsigned char& i) {
    float min, max, delta;

    min = minF(r, g, b);
    max = maxF(r, g, b);
    i = (unsigned char) (((float) (r + g + b)) / 3.f);

    delta = max - min;
    float fS;
    float fH;

    float v1 = -(((sqrt(6.f) * (float)r) / 6.) +
            ((sqrt(6.f) * (float)g) / 6.) +
            ((sqrt(6.f) * (float)b) / 3.));

    float v2 = ((float)r / sqrt(6.f)) - ((2. * (float)g) / sqrt(6.f));    

    fH = atan2(v2, v1);
    fH = radToDegrees(fH);

    if (fH < -360.)
        fH = -360.;

    if (fH < 0.)
        fH += 360.;

    if (fH > 360.)
        fH = 360.;    
    fS =  1 - 3 * ((float) min / (float) i);
    

    h = (unsigned int) fH;    
    s = (unsigned char) (fS * 255.f);
}

unsigned int* convertImagetoHSI(IplImage* image, int& height, int& width, int& channels,
        void (*convertFunc)(unsigned char, unsigned char, unsigned char, unsigned int&, unsigned char&, unsigned char&)) {
    height = image->height;
    width = image->width;
    channels = image->nChannels;
    unsigned int* retArr = new unsigned int[height * width * channels];

    unsigned char* data = (unsigned char*) image->imageData;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            unsigned char B = data[i * image->widthStep + j * channels + 0];
            unsigned char G = data[i * image->widthStep + j * channels + 1];
            unsigned char R = data[i * image->widthStep + j * channels + 2];

            unsigned int H;
            unsigned char S;
            unsigned char I;

            convertFunc(R, G, B, H, S, I);
            int index = i * width * channels + j * channels;

            retArr[index + 0] = H;
            retArr[index + 1] = (unsigned int) S;
            retArr[index + 2] = (unsigned int) I;
        }
    }

    return retArr;
}

unsigned char* simpleTsai(unsigned int* inputHSI, int height, int width, int channels) {
    float max = -1, min = FLT_MAX;
    float maxH = -1, minH = FLT_MAX;
    float maxI = -1, minI = FLT_MAX;
    unsigned char* retArr = new unsigned char[height * width];

    float maxVal = 360.f;// * 255.f;// + 1;
    float minVal = 0.f;
    float delta = maxVal - minVal;
    float segment = delta / 255;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width * channels + j * channels;

            unsigned int H = inputHSI[index + 0];
            unsigned int S = inputHSI[index + 1];
            unsigned int I = inputHSI[index + 2];

            if (maxH < H)
                maxH = H;
            if (minH > H)
                minH = H;

            if (maxI < I)
                maxI = I;
            if (minI > I)
                minI = I;

            //H = (unsigned int)((float)(255.f / 360.f) * (float)H);

            float ratio = (float) (H) / ((float) (I) + 1.f);

            if (max < ratio)
                max = ratio;
            if (min > ratio)
                min = ratio;

            ratio -= minVal;
            ratio /= segment;
            retArr[i * width + j] = (unsigned char) ratio;
        }
    }

    std::cout << max << "," << min << endl;
    std::cout << maxH << "," << minH << endl;
    std::cout << maxI << "," << minI << endl;
    return retArr;
}

IplImage* get8bitImage(unsigned char* input, int height, int width) {
    IplImage* image = cvCreateImage(cvSize(width, height), 8, 1);
    int wStep = image->widthStep;
    unsigned char* data = (unsigned char*) image->imageData;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data[i * wStep + j] = input[i * width + j];
        }
    }

    return image;
}

IplImage* binarize(IplImage* input) {
    IplImage* image = cvCreateImage(cvSize(input->width, input->height), input->depth, input->nChannels);

    double thresh = cvThreshold(input, image, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    return image;
}

IplImage* joinTwo(IplImage* src1, IplImage* src2){
    IplImage* image = cvCreateImage(cvSize(src1->width, src1->height), src1->depth, src1->nChannels);
    cvOr(src1, src2, image);
    return image;
}

int main(int argc, char **argv) {

    if (argc > 2) {
        char* path = argv[1];
        IplImage* image = cvLoadImage(path);
        if (image != 0) {
            int height, width, channels;
            unsigned int* hsi1 = convertImagetoHSI(image, height, width, channels, &RGBtoHSI_1);
            unsigned char* ratios1 = simpleTsai(hsi1, height, width, channels);
            IplImage* ratiosImage1 = get8bitImage(ratios1, height, width);            
            IplImage* binarized1 = binarize(ratiosImage1);
            
            unsigned int* hsi2 = convertImagetoHSI(image, height, width, channels, &RGBtoHSI_2);
            unsigned char* ratios2 = simpleTsai(hsi2, height, width, channels);
            IplImage* ratiosImage2 = get8bitImage(ratios2, height, width);            
            IplImage* binarized2 = binarize(ratiosImage2);
            
            IplImage* join = joinTwo(binarized1, binarized2);
            
            delete[] hsi1;
            delete[] ratios1;
            delete[] hsi2;
            delete[] ratios2;
            char* savePath = argv[2];
                                   
            cvSaveImage(savePath, join);
            
            cvReleaseImage(&join);
            cvReleaseImage(&binarized1);
            cvReleaseImage(&ratiosImage1);
            cvReleaseImage(&binarized2);
            cvReleaseImage(&ratiosImage2);
            cvReleaseImage(&image);
        }
    }    
}

