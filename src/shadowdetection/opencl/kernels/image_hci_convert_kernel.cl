/**
*find max of 3 values
*/
uchar maxF(uchar a, uchar b, uchar c) {
    uchar max = a;
    if (b > max)
        max = b;
    if (c > max)
        max = c;

    return max;
}

/**
*find min of 3 values
*/
uchar minF(uchar a, uchar b, uchar c) {
    uchar min = a;
    if (b < min)
        min = b;
    if (c < min)
        min = c;

    return min;
}

/**
*converts degrees to radians
*/
float radToDegrees(float radians) {
    const float PI_F = 3.14159265358979f;
    float oneRad = 180. / PI_F;
    return radians * oneRad;
}

/**
*converts RGB pixel to HSI first way
*/
uint3 convert1(uchar r, uchar g, uchar b)
{
    uint3 retVal;
    uchar min, max;

    min = minF(r, g, b);
    max = maxF(r, g, b);
    float i = convert_float_rtz(r + g + b) / 3.f;
    
    float fS;
    float fH;

    float v1 = 0.5f * (2.f * r - convert_float_rtz(g) - convert_float_rtz(b));
    float v2 = 0.5f * sqrt(3.f) * (convert_float_rtz(g) - convert_float_rtz(b));

    fH = atan2(v2, v1);
    fH = radToDegrees(fH);

    if (fH < -360.f)
        fH = -360.f;

    if (fH < 0.)
        fH += 360.f;

    if (fH > 360.f)
        fH = 360.f;    
    fS = 1.f - ((float)min / i);       
    if (fS > 1.f){
        fS = 1.f;
    }
    if (fS < 0.f){
        fS = 0.f;
    }
    fS *= 255.f;    
    retVal = (uint3)(convert_uint_rtz(fH), convert_uint_rtz(fS), convert_uint_rtz(i));
    return retVal;
}

/**
*converts RGB pixel to HSI second way
*/
uint3 convert2(uchar r, uchar g, uchar b)
{
    uint3 retVal;
    uchar min, max, delta;

    min = minF(r, g, b);
    max = maxF(r, g, b);
    float i = ((float)(r + g + b)) / 3.f;

    delta = max - min;
    float fS;
    float fH;

    float v1 = -(((sqrt(6.f) * (float)r) / 6.f) +
            ((sqrt(6.f) * (float)g) / 6.f) +
            ((sqrt(6.f) * (float)b) / 3.f));

    float v2 = ((float)r / sqrt(6.f)) - ((2.f * (float)g) / sqrt(6.f));    

    fH = atan2(v2, v1);
    fH = radToDegrees(fH);

    if (fH < -360.f)
        fH = -360.f;

    if (fH < 0.)
        fH += 360.f;

    if (fH > 360.)
        fH = 360.f;    
    fS = 1.f - ((float)min / i);       
    if (fS > 1.f){
        fS = 1.f;
    }
    if (fS < 0.f){
        fS = 0.f;
    }
    fS *= 255.f;
    
    retVal = (uint3)(convert_uint_rtz(fH), convert_uint_rtz(fS), convert_uint_rtz(i));    
    return retVal;
}

/**
*kernel functions wich converts RGB image to HSI image first way
*/
__kernel void image_hsi_convert1 (__global const uchar* input, __global uint* output, const uint width, const uint height, const uchar channels)                                                   
{
    /* get_global_id(0) returns the ID of the thread in execution.
    As many threads are launched at the same time, executing the same kernel,
    each one will receive a different ID, and consequently perform a different computation.*/
    const int index = get_global_id(0) * channels;    
    uint size = width * height * channels;

    if (index + 2 < size)
    {
        uchar r = input[index];
        uchar g = input[index + 1];
        uchar b = input[index + 2];        
        uint3 retPix = convert1(r, g, b);
        output[index] = retPix.x;
        output[index + 1] = retPix.y;
        output[index + 2] = retPix.z;
    }
}

/**
*kernel functions wich converts RGB image to HSI image second way
*/
__kernel void image_hsi_convert2 (__global const uchar* input, __global uint* output, 
                                 const uint width, const uint height, const uchar channels)                                                    
{
    /* get_global_id(0) returns the ID of the thread in execution.
    As many threads are launched at the same time, executing the same kernel,
    each one will receive a different ID, and consequently perform a different computation.*/
    const int index = get_global_id(0) * channels;    
    uint size = width * height * channels;

    if (index + 2 < size)
    {
        uchar r = input[index];
        uchar g = input[index + 1];
        uchar b = input[index + 2];        
        uint3 retPix = convert2(r, g, b);
        output[index] = retPix.x;
        output[index + 1] = retPix.y;
        output[index + 2] = retPix.z;
    }
}

/**
*kernel function performing simple tsai H vs I proportion on HSI image
*/
__kernel void image_simple_tsai (__global const uint* input, __global uchar* output, const uint width, const uint height, const uchar channels)                                                   
{
    /* get_global_id(0) returns the ID of the thread in execution.
    As many threads are launched at the same time, executing the same kernel,
    each one will receive a different ID, and consequently perform a different computation.*/
    const int index1 = get_global_id(0);
    const int index = index1 * channels;    
    uint size = width * height * channels;

    if (index + 2 < size)
    {
        float maxVal = 360.f;
        float minVal = 0.f;
        float delta = maxVal - minVal;
        float segment = delta / 255.f;
        
        uint h = input[index];
        uint s = input[index + 1];
        uint i = input[index + 2];        
        float ratio = (float)h / ((float)i + 1.f);
        ratio -= minVal;
        ratio /= segment;
        
        output[index1] = (uchar)ratio;        
    }
}