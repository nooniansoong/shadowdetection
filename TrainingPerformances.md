# Contents #
  * [Introduction](TrainingPerformances#Introduction.md)
  * [Data](TrainingPerformances#Data.md)
  * [Test Configuration](TrainingPerformances#Test_Hardware.md)
  * [Performance Diagram](TrainingPerformances#Performance_diagram.md)
  * [What has been done](TrainingPerformances#What_has_been_done.md)
  * [Limitations](TrainingPerformances#Limitations.md)
  * [Usage](TrainingPerformances#Usage.md)
# Introduction #
> For more detailed documentation and configuration settings please read: http://code.google.com/p/shadowdetection/wiki/ShadowDetection
# Data #
> Data used in training proces can be downloaded here: [Data File](http://www.mediafire.com/download/p18lggs51s9a29b/bigdata.svm.gz). File should be gunzipped before usage.
# Test\_Hardware #
> Same hardware used as in prediction performances, please see: [Test Configuration](http://code.google.com/p/shadowdetection/wiki/PredictionPerformances#Test_Hardware)
# Performance\_diagram #
> Training for C-SVC type and RBF kernel
> ![http://i969.photobucket.com/albums/ae180/markodjurovic/shadowdetection/performancesTraining_zpsf7f5edec.png](http://i969.photobucket.com/albums/ae180/markodjurovic/shadowdetection/performancesTraining_zpsf7f5edec.png)
# What\_has\_been\_done #
> Currently only parts of libsvm training process were parallelized (kernel values calculations). Opencl CPU and OpenCL GPU configurations are using same OpenCL code, but memory spaces are different. CPU configuration uses host memory space, and GPU configuration uses device memory space. Although in CPU configuration there are no (or minimal) memory coping, GPU is, in total, still faster.
# Limitations #
> Every record in data file must have same number of features, and there is no possibility to skip indexes. Indexes must be in ascending order. So this implementation is not fully compatible with native libsvm implementation.
**Currently only SVC and SVR types of are supported, one class SVM is in phase of native implementation with parallelization with OpenMP suggested by libsvm authors.**
# Usage #
> Build wanted build (_please see: https://code.google.com/p/shadowdetection/wiki/ShadowDetection#Command_line_), and start it with arguments: -training _data\_file output\_model\_file_. For example (if using AMD platform):
```
./dist/ReleaseOpenCL_AMD/GNU-Linux-x86/shadowdetection -training testData.svm testModel3.model
```
> Of course, training process can be used for any data until limitations are considered (_please see: [Limitations](TrainingPerformances#Limitations.md)_)