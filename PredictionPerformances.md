# Contents #
  * [Test Configuration](PredictionPerformances#Test_Hardware.md)
  * [Performance Diagram](PredictionPerformances#Performance_diagram.md)
  * [What has been done](PredictionPerformances#What_has_been_done.md)
  * [Limitations](PredictionPerformances#Limitations.md)
# Test\_Hardware #
  * AMD Athlon II 651 x4 Quad Core 3000MHz
  * 16GB DDR3 RAM 1333MHz (2 x 4GB + 1 x 8 GB, all modules Kingston Hyperx Blue)
  * AMD [R9](https://code.google.com/p/shadowdetection/source/detail?r=9) 270 2GB GDDR5
# Performance\_diagram #

![http://i969.photobucket.com/albums/ae180/markodjurovic/shadowdetection/performances_zpsa1090525.png](http://i969.photobucket.com/albums/ae180/markodjurovic/shadowdetection/performances_zpsa1090525.png)

> Yes, these values are true. On GPUs we have 28 seconds for image with dimensions 400x266, compare with 1012 seconds on CPUs. Both OpenCLCPU and OpenCLGPU configurations are using same OpenCL code.
# What\_has\_been\_done #

> In this what I've called openMP build/configurations, for openMP parallelization where used suggestions from authors of libsvm. So basically SVM kernel calculations were parallelized.

> In OpenCL builds / configurations predictions calculations were parallelized on image level. Also all calculations, but last, were done with single float precision.

> Basic unit in libsvm is svm\_node structure made of one double representing value, and one int representing node index. Signal for last node in nodes vector is index with value -1.

> I've changed that and before passing values to OpenCL all nodes' vectors lengths are precalculated, and only values are passed (not indexes).

> That allows usage of vector instead of scalar OpenCL types. Although both, NVIDIA and AMD, suggests usage of scalar types with new, GPUs, hardware, my experience tells me that in number of situations working with vector types is, in some cases, up to 5 times faster, and this is one of those cases.

> For example:
```
float kfunction_rbf(    const __global float* x, const size_t xLen,
                        const __global float* y, const size_t yLen,
                        float gamma){
    float sum = 0;    
    int i = 0;
    while (i < xLen){
        int remain = xLen - i;
        if (remain >= 8){
            float8 xv = vload8(i / 8, x);
            float8 yv = vload8(i / 8, y);
            float8 d = xv - yv;
            d *= d;
            sum += d.s0 + d.s1 + d.s2 + d.s3 + d.s4 + d.s5 + d.s6 + d.s7;
            i += 8; 
        }
        else if (remain >= 4){
            float4 xv = vload4(i / 4, x);
            float4 yv = vload4(i / 4, y);
            float4 d = xv - yv;            
            d *= d;            
            sum += d.x + d.y + d.z + d.w;
            i += 4;
        }
        else if (remain >= 2){
            float2 xv = vload2(i / 2, x);
            float2 yv = vload2(i / 2, y);
            float2 d = xv - yv;
            d *= d;            
            sum += d.x + d.y;
            i += 2;
        }
        else if (remain >= 1){
            float d = x[i]  - y[i];
            sum += d * d;
            i++;
        }        
    }
    return exp(-gamma * sum);
}
```

> is more than 3 times faster than
```
float kfunction_rbf(    const __global float* x, const size_t xLen,
                        const __global float* y, const size_t yLen,
                        float gamma){
    float sum = 0;
    for (int i = 0; i < xLen; i++){
       float d = x[i]  - y[i];
       sum += d * d;
    }
    return exp(-gamma * sum); 
} 
```

> Of course that performance gain is possible when numbers of nodes in nodes' array are aligned to numbers which allows more usage of vectorisation ((2 to power of n) + (2 to power of m)..., when m, n ... are > 1). Here number of elements in array is 13.

# Limitations #
> This implementation of libsvm is not always compatible with original implementation. Here every array must have same number of elements (both in training and prediction). Also in prediction input arrays and model arrays must have same number of elements (model is prediction model built in training process of SVM)