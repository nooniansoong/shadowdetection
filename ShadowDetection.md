|![http://i969.photobucket.com/albums/ae180/markodjurovic/shadowdetection/camel_zps55f6bcba.jpg](http://i969.photobucket.com/albums/ae180/markodjurovic/shadowdetection/camel_zps55f6bcba.jpg)|![http://i969.photobucket.com/albums/ae180/markodjurovic/shadowdetection/camelShadow_SVM_zpsc6e32273.jpg](http://i969.photobucket.com/albums/ae180/markodjurovic/shadowdetection/camelShadow_SVM_zpsc6e32273.jpg)|
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# Aditional info #

  * Project can be traced at: [ohloh.net](http://www.ohloh.net/p/shadowdetection), thanks guys for this :).

&lt;wiki:gadget url="http://www.ohloh.net/p/716824/widgets/project\_thin\_badge.xml" height="52" border="0"/&gt;

  * **Project uses modified libsvm project: http://www.csie.ntu.edu.tw/~cjlin/libsvm/, thanks to authors on this great and reliable library.**
# Current version #
> Current version is: 0.7.0

> About version 0.2.0 please see: [ver0.2.0](http://code.google.com/p/shadowdetection/wiki/Version020)
> About version 0.6.0 please see: [ver0.6.0](https://code.google.com/p/shadowdetection/wiki/Version060)

# Contents #
  * [News](ShadowDetection#News.md)
  * [Requirements](ShadowDetection#Requirements.md)
  * [Config](ShadowDetection#Config.md)
  * [Future](ShadowDetection#Future.md)
  * [Usage](ShadowDetection#Usage.md)
  * [Batch file](ShadowDetection#Batch_file.md)
  * [Optimisation](ShadowDetection#Optimisation.md)
  * [Results](ShadowDetection#Results.md)
  * [What has been done](ShadowDetection#What_has_been_done.md)
  * [Make your model](ShadowDetection#Make_your_model.md)
  * [Expand with new features](ShadowDetection#Expand_with_new_features.md)
  * [Contact](ShadowDetection#Contact.md)
# News #
  * **Unfortunately no more access to Tesla card, so any help of providing testing on Tesla cards will be great.**
  * Mac OSX builds abandoned for now.
  * Version 0.7.0
    1. Updated documenation about config file: [Config](ShadowDetection#Config.md) (09/13/14)
    1. Parallelized regression pediction (09/13/14)
    1. Changed ImageParameters interface (09/07/14)
    1. Switched to C++11 (09/06/14)
    1. Reorganized configuration file (09/06/14)
    1. More possibilities to extend functionality (09/06/14)
    1. Added memory tracking in debug mode (09/06/14)
    1. Added RTTI (09/06/14)
    1. Removed usage of deprecated collections (09/06/14)
    1. Possibility to switch functionality to something other than shadow detection (by implementing new classes) (09/06/14)
    1. Removed some memory leaks (09/06/14)
  * Version 0.6.0
    1. Changed root node of config file (09/01/14)
    1. Fixed issue with OpenCl initialisation in training process (08/26/14)
    1. New prediction based on logistic regression (08/24/14)
    1. Fixed false detected shadow pixels on lightening areas (08/24/14)
    1. Partially fixed false detected in sky areas (08/24/14)
    1. Sky detection algorithm (08/24/14)
    1. Support for making prediction models (08/24/14)
    1. New tools in order to make new models (08/24/14)
    1. First version of API/Framework for general pixel categorsation (08/24/14)
    1. Abandoned Mac OSX support for now, because of issues with OpenCL (08/24/14)
    1. New model for SVM prediction (08/24/14) (_Please see: [SVM Prediction](ShadowDetection#SVM_Prediction.md) for details_).
  * Version 0.2.0
    1. Fixed issues with NU-SVR prediction (07/24/14)
    1. For now paused with svm training process parallelization (07/24/14)
    1. Added support for SVR trainings (07/14/14)
    1. Done more svm training optimisation (07/12/14)
    1. Fixed memory release issue (07/12/14)
    1. Added comparation performances char (_see: [Optimisation](ShadowDetection#Optimisation.md)_) (07/09/14)
    1. Finished optimizing for svm predict. Target hardware was GPU. Results very good, can be seen here: [Performances](http://code.google.com/p/shadowdetection/wiki/PredictionPerformances) (07/08/14)
    1. Fixed issue with non OpenCL builds (07/08/14)
    1. Uploaded model file (_see: [SVM Predictions](http://code.google.com/p/shadowdetection/wiki/ShadowDetection#SVM_Prediction) for instructions_) (07/02/14)
    1. Ability to use svm prediction for more accurate results, see: [Results](http://code.google.com/p/shadowdetection/wiki/Results) (07/02/14)
    1. Parallelized svm prediction (on image level) with OpenCL (still not optimal )(07/01/14)
    1. Removed some memory leaks (07/01/14)
    1. Started using JAVA for external tools (07/01/14)
    1. Started using first "dummy" version of memory manager (07/01/14)
    1. Added tool for making training sets (07/01/14)
    1. For non OpenCL builds some parts of svm parallelized with OpenMP (07/01/14)
  * Version 0.1.0
    1. Now works with AMD GPUs, tested on [R9](https://code.google.com/p/shadowdetection/source/detail?r=9) 270 (06/24/14)
    1. Libsvm training OpenCL parallelization stopped on current level, will be continued later (06/24/14)
    1. Tested with Intel CPUs (06/18/14)
    1. Removed OpenMP support from Mac and OpenCL builds (06/14/14)
    1. Fixed issue with save binary on Mac (06/08/14)
    1. Fixed load precompiled binary for AMD CPUs (06/07/14)
    1. Added support for MacOSX (06/07/14)
    1. OpenCL processing switched to OpenCV2 (06/06/14)
    1. Fixed issue with multiple OpenCL platforms installed (06/04/14)
    1. Added OpenCL support for AMD CPUs (06/04/14)
    1. Added OpenCL processing support (see [Requirements](ShadowDetection#Requirements.md))
    1. Added batch processing (see [Config](ShadowDetection#Config.md))
    1. Added configuration (see [Config](ShadowDetection#Config.md))
# Requirements #
> ## Supported operating systems ##
    * Linux 64 bit
> ## Supported CPUs ##
    * x86\_64
> ## Supported image formats ##
    * jpeg
    * tiff
    * png
    * bmp
> ## Required libraries ##
    * libjpeg and dev ( [libjpeg](http://libjpeg.sourceforge.net) )
    * libtiff and dev ( [libtiff](http://www.libtiff.org/) )
    * libpng and dev  ( [libpng](http://www.libpng.org/pub/png/libpng.html) )
    * OpenCV 2.4.9 or above (if openCV is installed with different prefix than /usr/local then must set adequate include and lib paths)
    * make
    * gcc version 4.4.8 or above
    * g++
    * libgomp (_for linux systems_)
> > _Notice: Some libraries have them own requirements_

> ## SVM Prediction ##
    * For SVM prediction you need model file. It candownloaded from: [model](http://www.mediafire.com/download/3dbj738o4wi4z4g/bigModel_RemovedThree.model.gz). Gunzip it to root dir of project. For enabling/disabling SVM prediction see config section: [Config](ShadowDetection#Config.md)

> _Notice: Old model file is not compatible with version 0.6.0_
> ## OpenCL ##
> > ### Supported device hardware ###
      * NVIDIA GeForce 600 series or above
      * NVIDIA Tesla K series
      * AMD CPUs
      * AMD GPUs series [R9](https://code.google.com/p/shadowdetection/source/detail?r=9)
      * Intel CPUs (works even with AMD-APP-SDK-v2.9)
> > > > _Notice: May work on some other NVIDIA cards than specified but it is not tested, for AMD cards also may work with earlier models but not tested_

> > ### Supported host hardware ###
      * Intel CPUs
      * AMD CPUs
> > ### Required software ###
> > > #### NVIDIA GPUs ####
        * Latest NVIDIA OpenCL capable drivers
        * NVIDIA CUDA Toolkit v6 (if cuda toolkit is installed at different location than /usr/local/cuda-6.0/ then must set adequate lib and include paths)
> > > #### AMD and Intel CPUs ####
        * AMD-APP-SDK-v2.9 (if SDK installed at different location than /opt/AMDAPP/ then must set adequate lib and include paths)
> > > #### AMD GPUs ####
        * AMD Catalyst drivers ver 14.10 or above
      * OpenCV built with WITH\_OPENCL turned ON
> > > > ![http://i969.photobucket.com/albums/ae180/markodjurovic/shadowdetection/opencv_zpsb0a30251.png](http://i969.photobucket.com/albums/ae180/markodjurovic/shadowdetection/opencv_zpsb0a30251.png)

> ## Java tools ##
    * JDK 1.7 or above
    * ant and and-optional packages ver 1.9.3 or above. Project ant page: http://ant.apache.org/
# Config #
Please see: [Configuration file](http://code.google.com/p/shadowdetection/wiki/ConfigurationFile)
# Usage #
> ## Build ##
> > ### NetBeans IDE ###
      * This is netbeans 8.0 project. If you are using this version or above just open it from File->Open Project. If required dependecies are not installed at default locations, you got to set adequate include and lib dirs in project properties.
      * For Java tools project open project located in root of base project with name ShadowDetectionTools. Treat it as any other Java project.


> ### Command line ###
    * Configurations are: Debug, Release, DebugOpenCL, ReleaseOpenCL, DebugOpenCL\_AMD, ReleaseOpenCL\_AMD.
> > > make usage:
```
Makefile Usage:"
	make [CONF=<CONFIGURATION>] [SUB=no] build"
	make [CONF=<CONFIGURATION>] [SUB=no] clean"
	make [SUB=no] clobber"
	make [SUB=no] all"
	make help"
```
> > > OpenCL builds with no extension (DebugOpenCL and ReleaseOpenCL) are dedicated to NVIDIA platform
    * If required dependecies are not insatlled at default locations, you got to set adequate lib and include dirs in make files.
> > > #### Java tools ####
        * Go to ShadowDetectionTools dir
        * Build with ant (just type "ant" with no quotes)

> ## Use instructions ##
    * After successful build you will have appropriate executables.
    * OpenCL executables are using openCL in processing, other not.
    * Set configuration file ( see [Config](ShadowDetection#Config.md) )
    * Call executable from root dir:
      1. If use batch with parameter which indicates batch processing file (see [Batch file](ShadowDetection#Batch_file.md) ). Example:
```
./dist/ReleaseOpenCL/GNU+CUDA-Linux-x86/shadowdetection list.csv
```
      1. If not using batch processing with input and output paramaters. Example:
```
./dist/ReleaseOpenCL/GNU+CUDA-Linux-x86/shadowdetection 0.jpg out.jpg
```
    * Additional parameters -help and -list. Help will provide link to this wiki page. For "openCL" builds -list will list all platforms and devices, example:
```
$ ./dist/ReleaseOpenCL_AMD/GNU-Linux-x86/shadowdetection -list
Found 2 platforms.
=============
Platform name: AMD Accelerated Parallel Processing
Found 1 devices
Device 0 name: AMD Athlon(tm) II X4 651 Quad-Core Processor
=============
Platform name: NVIDIA CUDA
Error: clGetDeviceIDs(-1)
Platform not supported by this build
SHADOW_OTHER clGetDeviceIDs
=============
```

# Batch file #
Batch file is text file composed of lines. Each line has TAB separated values for input and output files. Example:
```
0.jpg	shadow01.jpg
5.jpg	shadow51.jpg
camel.jpg	shadowcamel1.jpg
deca.png	shadowdeca1.jpg
sh.jpg	shadowsh1.jpg
```
# Future #
> ## Done ##
    * ~~Complete switch to OpenCV C++ API~~
    * ~~MacOSX support~~
    * ~~Use precompiled openCL kernels~~
    * ~~Support to use AMD CPUs in openCL processing~~
    * ~~Parallelize libsvm training with OpenCL~~
    * ~~Support to use Intel CPUs in openCL processing~~
    * ~~AMD graphic cards support~~
    * ~~Parallelize libsvm predict on image level~~
    * ~~New options in config file due to add possibility to configure libsvm~~
    * ~~Improve accuracy with SVM~~
    * ~~Provide model file~~
    * ~~Switch to single float precision in svm prediction~~
    * ~~Update documentation~~
    * ~~Add checks for start parameters~~
    * ~~Performance comparation charts (parallelized vs. non-parallelized)~~
    * ~~Improve shadow detection with new techniques~~
    * ~~More API interfaces which will allow to extend functionality.~~
    * ~~RTTI (in version 1.0.0)~~
    * ~~Smart pointers (in version 1.0.0)~~
    * ~~General pixel categorisation framework (in version 1.0.0)~~
    * ~~Parallel "regression prediction"~~
> ## Very soon ##
    * How to (do something other than shadow detection) example
    * API documentation (in version 1.0.0)
> ## Soon ##
    * configure script (in version 1.0.0)
    * Switch project to "general pixel categorisation framework". (from version 1.0.0)
> ## Later ##
    * Windows and Visual Studio support.
    * Return Mac OSX builds.
    * Improve accuracy with machine learning (in version 2.0.0)
# Optimisation #
> For performances and explanantion of optimisation techniques please see: [Prediction Performances](http://code.google.com/p/shadowdetection/wiki/PredictionPerformances)
# Results #
> Please see: [Results](http://code.google.com/p/shadowdetection/wiki/Results)
# What\_has\_been\_done #
> Please see: [What has been done](http://code.google.com/p/shadowdetection/wiki/WhatHasBeenDone)
# Make\_your\_model #
> Please see: [Make your model](http://code.google.com/p/shadowdetection/wiki/MakeYourModel)
# Expand\_with\_new\_features #
> Please see: [Expand with new features](http://code.google.com/p/shadowdetection/wiki/ExpandWithNewFeatures)
# Contact #

> For any questions, suggestions, issues notifications, etc. please contact markodjurovic@yahoo.com