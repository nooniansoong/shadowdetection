# Content #
  * [Why I want to do this](MakeYourModel#Why?.md)
  * [First steps](MakeYourModel#First_steps.md)
  * [Build SVM model](MakeYourModel#Build_SVM_model.md)
  * [Get regression coefficients with R package](MakeYourModel#Get_regression_coefficients_with_R.md)
  * [Finally (read this first :) )](MakeYourModel#Finally.md)
# Why? #
> If you want to analyze only one type of digital images (for example some parking lot, only beaches, nature in mauntains, images made from air of specified area, ...), then probably you want to have your own model. Ideal case will include that all images was/will be made with same camera on same location or similar enough locations
# First\_steps #
> ## Prepare training set ##
    * Choose images you want to use as training images and put them in one dir (for example imagesResized)
    * Create batch file with command like this:
```
java -jar ShadowDetectionTools/dist/ShadowDetectionTools.jar -makelist imagesResized/ imagesResized.csv
```
    * Go to configuration file, find Prediction section and set property "usePrediction" to false
```
...
<Prediction>            
    <!-- true, false -->
    <usePrediction>
       false
    </usePrediction>
    <!--SVM for support vector machine, REG for regresion-->
    <predictionType>
       REG
    </predictionType>            
</Prediction>
...
```
    * Also set use batch to true:
```
...
    <process>
        <!-- true, false -->
        <UseBatch>
            true
        </UseBatch>
...
```
    * This will start shadow prediction using only basic algorithm and restriction defined connected with sky areas and lightness properties, so do it:
```
./dist/ReleaseOpenCL_AMD/GNU-Linux-x86/shadowdetection imagesResized.csv
```
    * After that every image will have them own pair with "Shadow" suffix in name.
    * Open batch file for edit (in some text editor and remove every pair where shadows didn't detected properly i.e. keep only images with least count of false detected pixels.
    * In configuration file choose will you have equal distribution of two classes, or not. Section "Training", here I want to have equal distribution. Which is better depends of situation, and this tutorial is not about that:
```
...
        <Training>
            <!-- true, false -->
            <distribute0and1>
                true
            </distribute0and1>
        </Training>
...
```
    * Make training set with something like this:
```
./dist/ReleaseOpenCL_AMD/GNU-Linux-x86/shadowdetection -makeset imagesResized.csv pixelData.svm
```
    * Content of result file should look something like this:
```
shadowdetection$ head pixelData.svm 
0 1:0.164706 2:0.788235 3:0.0142119 4:0.0030253 5:0.000815376 6:0.72549 7:0.235294 8:0.00328554 9:0.0100182 10:0.0118933 11:0.788235 12:0.680392
1 1:0.270588 2:0.360784 3:0.00888889 4:0.00669056 5:0.00290955 6:0.313726 7:0.156863 8:0.00768176 9:0.0151762 10:0.00765184 11:0.360784 12:0.276471
0 1:0.0705882 2:0.94902 3:0.0315789 4:0.00246914 5:0.000290487 6:0.917647 7:0.396078 8:0.00255319 9:0.00588235 10:0.00899654 11:0.94902 12:0.896078
1 1:0.278431 2:0.364706 3:0.00864198 4:0.00661939 5:0.00296204 6:0.313726 7:0.160784 8:0.00768176 9:0.0148148 10:0.00746965 11:0.364706 12:0.276471
0 1:0.0627451 2:0.941176 3:0.0346405 4:0.00244352 5:0.000260353 6:0.913725 7:0.333333 8:0.00251662 9:0.00684755 10:0.0106247 11:0.941176 12:0.896078
1 1:0.262745 2:0.356863 3:0.00931373 4:0.00688406 5:0.00285592 6:0.309804 7:0.152941 8:0.00791667 9:0.0158333 10:0.0077451 11:0.356863 12:0.272549
0 1:0.0627451 2:0.945098 3:0.0346405 4:0.00243343 5:0.000259277 6:0.917647 7:0.34902 8:0.00250591 9:0.00654321 10:0.0101961 11:0.945098 12:0.9
1 1:0.298039 2:0.329412 3:0.00822511 4:0.00745098 5:0.00350634 6:0.282353 7:0.176471 8:0.0086758 9:0.0137681 10:0.00613811 11:0.329412 12:0.241176
0 1:0.0627451 2:0.94902 3:0.0346405 4:0.00242341 5:0.00025821 6:0.921569 7:0.364706 8:0.00249529 9:0.00626478 10:0.00980392 11:0.94902 12:0.903922
1 1:0.270588 2:0.360784 3:0.00904762 4:0.00681004 5:0.00290955 6:0.313726 7:0.156863 8:0.00781893 9:0.0154472 10:0.00765184 11:0.360784 12:0.272549
```
    * Now everything is ready for making either SVM either regression based model
# Build\_SVM\_model #
  * In config file choose svm and kernel type you want to use for example
```
        <svm>
            <!-- full file name -->
            <modelFile>
                bigModel_RemovedThree.model
            </modelFile>            
            <!-- 0=C-SVC, 1=nu-SVC, 2=one-class, 3=epsilon-SVR, 4=nu-SVR -->
            <svm_type>
                0
            </svm_type>
            <!-- 0=linear, 1=polynomial, 2=radial basis function, 3=sigmoid, 4-not supported -->
            <kernel_type>
                2
            </kernel_type>
        </svm>
```
  * Probably is best to use some SVC type and RBF kernel
  * Start training process with:
```
./dist/ReleaseOpenCL_AMD/GNU-Linux-x86/shadowdetection -training pixelData.svm pixelDataModel.model
```
  * In this example pixelDataModel.model is model file
# Get\_regression\_coefficients\_with\_R #
> For getting regression coefficients you will need R project installed: http://www.r-project.org/. In generally R is great software which with them tools can help you to analyse data, visualise data, determine if some data will be good to use as model, ....
  * Convert data to CSV format:
```
java -jar ShadowDetectionTools/dist/ShadowDetectionTools.jar -convertsvm pixelData.svm pixelData.csv
```
  * Data will change format to something like this:
```
shadowdetection$ head pixelData.csv 
0       0.164706        0.788235        0.0142119       0.0030253       0.000815376     0.72549 0.235294        0.00328554      0.0100182       0.0118933       0.788235        0.680392
1       0.270588        0.360784        0.00888889      0.00669056      0.00290955      0.313726        0.156863        0.00768176      0.0151762       0.00765184      0.360784        0.276471
0       0.0705882       0.94902 0.0315789       0.00246914      0.000290487     0.917647        0.396078        0.00255319      0.00588235      0.00899654      0.94902 0.896078
1       0.278431        0.364706        0.00864198      0.00661939      0.00296204      0.313726        0.160784        0.00768176      0.0148148       0.00746965      0.364706        0.276471
0       0.0627451       0.941176        0.0346405       0.00244352      0.000260353     0.913725        0.333333        0.00251662      0.00684755      0.0106247       0.941176        0.896078
1       0.262745        0.356863        0.00931373      0.00688406      0.00285592      0.309804        0.152941        0.00791667      0.0158333       0.0077451       0.356863        0.272549
0       0.0627451       0.945098        0.0346405       0.00243343      0.000259277     0.917647        0.34902 0.00250591      0.00654321      0.0101961       0.945098        0.9
1       0.298039        0.329412        0.00822511      0.00745098      0.00350634      0.282353        0.176471        0.0086758       0.0137681       0.00613811      0.329412        0.241176
0       0.0627451       0.94902 0.0346405       0.00242341      0.00025821      0.921569        0.364706        0.00249529      0.00626478      0.00980392      0.94902 0.903922
1       0.270588        0.360784        0.00904762      0.00681004      0.00290955      0.313726        0.156863        0.00781893      0.0154472       0.00765184      0.360784        0.272549
```
  * Start R environment with R command:
```
shadowdetection$ R

R version 3.0.2 (2013-09-25) -- "Frisbee Sailing"
Copyright (C) 2013 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

> 
```
  * Load data with:
```
myData <- read.csv(file="pixelData.csv", head=FALSE, sep="\t")
```
  * Do logistic regression of (for example poisson family):
```
glmPoisson <- glm(V1 ~., family = poisson(), data = myData)
```
  * When you get summary of model you will get coefficients too, and info about them, something like this:
```
> summary(glmPoisson)
```
```
Coefficients:
              Estimate Std. Error z value Pr(>|z|)    
(Intercept) -0.2793042  0.0182126 -15.336  < 2e-16 ***
V2          -0.0003959  0.0760243  -0.005  0.99585    
V3           2.1434379  0.1731929  12.376  < 2e-16 ***
V4          -3.5817859  2.2241620  -1.610  0.10731    
V5          -3.7359079  0.8278695  -4.513  6.4e-06 ***
V6          -1.3690708  0.1121611 -12.206  < 2e-16 ***
V7          -4.8352408  0.3800713 -12.722  < 2e-16 ***
V8           0.6896627  0.0674030  10.232  < 2e-16 ***
V9           1.1147119  0.5510998   2.023  0.04310 *  
V10         11.1865999  1.2566664   8.902  < 2e-16 ***
V11         -0.7745016  0.2908248  -2.663  0.00774 ** 
V12          1.6306103  0.1247026  13.076  < 2e-16 ***
V13         -3.0928828  0.2058099 -15.028  < 2e-16 ***
```
  * Didn't paste whole output but this is not so good model :).
  * Edit coefficients values in config file, following rule that V(N) is coefNo(N-1), for example coefNo2 should have value of V3,.... Intercept is Intercept
  * **Notice: Poisson family here were used just as an example.** More examples can see in src/R/getCoefs.R file
> ## Adjust borderValue ##
    * In most cases borderValue value is not 0.5 (which is not strange when we look formula which is used in prediction)
      1. So when you want to adjust border value turn prediction ON in config file.
      1. Set prediction type to "REG"
      1. Test with some test set
      1. Check results
      1. If satisfied goto step 8
      1. Change border value
      1. goto step 3
      1. you are done :)
> # Finally #
    * When finish everything, don't forget to turn prediction on again, set adequate values to properties in configuration file, ...
    * Do not use to many records in training process. 200K to 600K records should be enough. Better to use less, but more quality records, then more but less quality records. First you will not get better accuracy (in opposite) and second you will prolong processing time both in training and prediction.
    * If you want to reduce number of records by removing every Nth pair of records call something like this (this will remove every second pair form pixelData.svm file):
```
java -jar ShadowDetectionTools/dist/ShadowDetectionTools.jar -removepairs 2 pixelData.svm pixelData_Reduced.svm
```
> > ## How to recognize good data ##
> > > I can't give you answer because of two main reasons:
      * I am not sure about that :)
      * About things I know, I can't give brief answer, so research :)