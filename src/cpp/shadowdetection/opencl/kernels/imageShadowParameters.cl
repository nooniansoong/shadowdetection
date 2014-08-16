uchar3 getPixel(__global uchar* image, const int index){    
    __global uchar* pos = image + index * 3;
    uchar3 pixel = (uchar3)(pos[0], pos[1], pos[2]);
    return pixel;
}

void processHSV(__global float* currRow, uchar3 pix){
    currRow[0] = (float) pix.y / 255.f;
    currRow[0] = clamp(currRow[0], 0.f, 1.f);
    currRow[1] = (float) pix.z / 255.f;
    currRow[1] = clamp(currRow[1], 0.f, 1.f);
    currRow[2] = (float) pix.x / (float) (pix.y + 1);
    currRow[2] /= 180.f;
    currRow[2] = clamp(currRow[2], 0.f, 1.f);
    currRow[3] = (float) pix.x / (float) (pix.z + 1);
    currRow[3] /= 180.f;
    currRow[3] = clamp(currRow[3], 0.f, 1.f);
    currRow[4] = (float) pix.y / (float) (pix.z + 1);
    currRow[4] /= 255.f;
    currRow[4] = clamp(currRow[4], 0.f, 1.f);
}

void processHLS(__global float* currRow, uchar3 pix){
    currRow[5] = (float) pix.y / 255.f;
    currRow[5] = clamp(currRow[5], 0.f, 1.f);
    currRow[6] = (float) pix.z / 255.f;
    currRow[6] = clamp(currRow[6], 0.f, 1.f);
    currRow[7] = (float) pix.x / (float) (pix.y + 1);
    currRow[7] /= 180.f;
    currRow[7] = clamp(currRow[7], 0.f, 1.f);
    currRow[8] = (float) pix.x / (float) (pix.z + 1);
    currRow[8] /= 180.f;
    currRow[8] = clamp(currRow[8], 0.f, 1.f);
    currRow[9] = (float) pix.y / (float) (pix.z + 1);
    currRow[9] /= 255.f;
    currRow[9] = clamp(currRow[9], 0.f, 1.f);
}

void processBGR(__global float* currRow, uchar3 pix){
    currRow[10] = (float)pix.x / 255.f;
    currRow[10] = clamp(currRow[10], 0.f, 1.f);
    currRow[11] = (float)(pix.y + pix.z) / (255.f + 255.f);
    currRow[11] = clamp(currRow[11], 0.f, 1.f);
}

__kernel void imageShadowParameters(__global float* retMatrix, const uint numOfParameters,
                                    __global uchar* originalImage, __global uchar* hsvImage,
                                    __global uchar* hlsImage, const uint numOfPixels){
    int currIndex = get_global_id(0);
    int maxNum = numOfPixels;
    if (currIndex < maxNum){
        uchar3 picHSV = getPixel(hsvImage, currIndex);
        uchar3 picHLS = getPixel(hlsImage, currIndex);
        uchar3 picRGB = getPixel(originalImage, currIndex);
        
        __global float* currRow = retMatrix + currIndex * numOfParameters;
        processHSV(currRow, picHSV);
        processHLS(currRow, picHLS);
        processBGR(currRow, picRGB);
    }
}
