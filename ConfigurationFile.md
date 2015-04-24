# File #

Configuration file is: ShadowDetectionConfig.xml and looks like this:
```
<?xml version="1.0" encoding="utf-8"?>

<settings>
    <general>
        <!-- true, false -->
        <UseBatch>
            true
        </UseBatch>
        
        <Training>
            <!-- true, false -->
            <distribute0and1>
                true
            </distribute0and1>
            <svm>
                <!-- 0=C-SVC, 1=nu-SVC, 2=one-class, 3=epsilon-SVR, 4=nu-SVR -->
                <svm_type>
                    0
                </svm_type>
                <!-- 0=linear, 1=polynomial, 2=radial basis function, 3=sigmoid, 4-not supported -->
                <kernel_type>
                    2
                </kernel_type>
            </svm>
        </Training>
        
        <Prediction>            
            <!-- true, false -->
            <usePrediction>
                true
            </usePrediction>
            
            <!--SVM for support vector machine, REG for regresion-->
            <predictionClass>
                core::util::prediction::regression::RegressionPredict
            </predictionClass>
            
            <svm>
                <!-- full file name -->
                <modelFile>
                    bigModel_RemovedThree.model
                </modelFile> 
            </svm>
            
            <regression>
                <!-- number of coefficients used in regression classification, do not involve intercept -->
                <coefNum>
                    12
                </coefNum>
                <!-- numerical value of coefficient -->
                <coefNo1>
                    -0.78317 
                </coefNo1>
                <coefNo2>
                    -0.32998 
                </coefNo2>
                <coefNo3>
                    6.30192 
                </coefNo3>
                <coefNo4>
                    8.70635
                </coefNo4>
                <coefNo5>
                    -3.11831 
                </coefNo5>
                <coefNo6>
                    -2.51965 
                </coefNo6>
                <coefNo7>
                    1.49102 
                </coefNo7>
                <coefNo8>
                    -2.45055 
                </coefNo8>
                <coefNo9>
                    5.74217 
                </coefNo9>
                <coefNo10>
                    -6.24504 
                </coefNo10>
                <coefNo11>
                    1.18127
                </coefNo11>
                <coefNo12>
                    -0.36501
                </coefNo12>            
                <Intercept>
                    -0.52834
                </Intercept>
                <!-- formula used for shadow detection here is: 
                1 / (1 + exp(-alpha)),
                where alpha = Intercept + coefNo1 * parameter1 + ... + coefNocoefNum * parametercoefNum
                If result is > then border value then pixel is candidate for shadow -->
                <borderValue>
                    0.39
                </borderValue>
            </regression>
            <parametersClass>
                shadowdetection::tools::image::ImageShadowParameters
            </parametersClass>            
        </Prediction>
        
        <openCL>            
            <!-- true, false -->
            <UsePrecompiledKernels>
                false
            </UsePrecompiledKernels>
            <!-- index of platform -->
            <platformid>
                0
            </platformid>
            <!-- index of device -->
            <deviceid>
                0
            </deviceid>    
        </openCL>
        
        <openMP>
            <!-- openMP thread number -->
            <threadNum>
                4
            </threadNum>
        </openMP>
        
        <!-- something like simple reflection, do not touch for now -->
        <classes>
            <!-- general processor class -->
            <processorClass>
                shadowdetection::process::ShadowDetectionProcessor
            </processorClass>
            
            <core::opencl::regression::OpenCLRegressionPredict>
                <kernels>
                    <kernelCount>
                        1
                    </kernelCount>
                    <kernelNo0>
                        predict
                    </kernelNo0>                    
                </kernels>
                <programs>
                    <programFile>
                        RegressionPredict
                    </programFile>
                    <rootDir>
                        src/cpp/core/opencl/regression/kernels/
                    </rootDir>
                </programs>
            </core::opencl::regression::OpenCLRegressionPredict>
            
            <core::opencl::libsvm::OpenCLToolsTrain>
                <kernels>
                    <kernelCount>
                        3
                    </kernelCount>
                    <kernelNo0>
                        svcQgetQ
                    </kernelNo0>
                    <kernelNo1>
                        svrQgetQ
                    </kernelNo1>
                    <kernelNo2>
                        selectWorkingSet
                    </kernelNo2>
                </kernels>
                <programs>
                    <programFile>
                        libSvmTrain
                    </programFile>
                    <rootDir>
                        src/cpp/core/opencl/libsvm/kernels/
                    </rootDir>
                </programs>
            </core::opencl::libsvm::OpenCLToolsTrain>
            
            <core::opencl::libsvm::OpenCLToolsPredict>
                <kernels>
                    <kernelCount>
                        1
                    </kernelCount>
                    <kernelNo0>
                        predict
                    </kernelNo0>                    
                </kernels>
                <programs>
                    <programFile>
                        libSvmPredict
                    </programFile>
                    <rootDir>
                        src/cpp/core/opencl/libsvm/kernels/
                    </rootDir>
                </programs>
            </core::opencl::libsvm::OpenCLToolsPredict>
            
            <shadowdetection::opencl::OpenclTools>
                <kernels>
                    <kernelCount>
                        3
                    </kernelCount>
                    <kernelNo0>
                        image_hsi_convert1
                    </kernelNo0>
                    <kernelNo1>
                        image_hsi_convert2
                    </kernelNo1>
                    <kernelNo2>
                        image_simple_tsai
                    </kernelNo2>
                </kernels>
                <programs>
                    <programFile>
                        image_hci_convert_kernel
                    </programFile>
                    <rootDir>
                        src/cpp/shadowdetection/opencl/kernels/
                    </rootDir>
                </programs>
            </shadowdetection::opencl::OpenclTools>
            
            <shadowdetection::opencl::OpenCLImageParameters>
                <kernels>
                    <kernelCount>
                        1
                    </kernelCount>
                    <kernelNo0>
                        imageShadowParameters
                    </kernelNo0>                    
                </kernels>
                <programs>
                    <programFile>
                        imageShadowParameters
                    </programFile>
                    <rootDir>
                        src/cpp/shadowdetection/opencl/kernels/
                    </rootDir>
                </programs>
            </shadowdetection::opencl::OpenCLImageParameters>
            
        </classes>
                
    </general>
    <shadowDetection>
        <!-- tells whether to use or not thresholds bellow -->
        <useThresholds>
            true;
        </useThresholds>
        
        <!-- threshold values for results correction -->            
        <Thresholds>                                    
            <!-- maximum lightness value for shadow pixel. HSL color space.
            If pixel has bigger value then specified it can not be shadow pixel-->                
            <lValue>
                150
            </lValue>
            <!-- true, false -->            
        </Thresholds>
        
        <!-- tells whether to use or not correction on sky areas in shadow
        detection process -->
        <useSkyDetection>
            true
        </useSkyDetection>
        
    </shadowDetection>
    <skyDetection>
        <!-- sky detection threshold values 
        If useSkyDetection is true pixel detected as sky pixel then it can not be shadow pixel-->
        <Thresholds>                        
            <!-- maximum R channel value for sky pixel -->
            <rValue>
                51
            </rValue>
            <!-- minimum B channel value for sky pixel -->
            <bValue>
                110
            </bValue>
            <!-- minimum lightness value for sky pixel -->
            <lValue>
                50
            </lValue>            
        </Thresholds>
    </skyDetection>    
</settings>
```

# Variables #

> "settings.general.UseBatch" variable should have true/false value.
    * If it is true then specify file with list of images should be processed ( look [Usage](ShadowDetection#Usage.md) )
    * If it is false then specify input image file and output file with extension

> "settings.general.Training.distribute0and1" variable specifies if classes in svm training will be distributed or not

> "settings.general.Prediction.svm.modelFile" path to model file if svm prediction is used

> "process.Prediction.usePrediction" specifies whether or not to use svm prediction in detection

> "settings.general.Prediction.predictionClass" is classID of class used in prediction. For regression prediction ClassID should be: core::util::prediction::regression::RegressionPredict. For svm prediction ClassID should be: core::util::prediction::svm::SvmPredict

> "settings.general.Training.distribute0and1" tells to the make training files process whether or not to equally distribute 0s and 1s.

> "process.Thresholds.shadow.lValue" specifies maximum lightness value for shadow pixel. If lightness value of candidate pixel is larger then specified in this property, pixel will not be declared as shadow pixel.

> "shadowDetection.Thresholds.useThresh" specifies if to use threshold reduction

> "skyDetection.Thresholds.rValue"
> "skyDetection.Thresholds.bValue"
> "skyDetection.Thresholds.lValue" are thresholds in sky detection process. Formula used is:
```
if ((R <= rThresh || R <= B / 3U) && 
    (G >= B / 6U) && 
     G <= B && B >= bThresh && 
     L >= lThresh)
```
> where R, G, B, L are channel values of current pixel, and rThresh, bThresh, lThresh are values specified in condiguration file. If pixel "passes this test" this doesn't mean that pixel will be treated as sky pixels, with this pass it only become candidate for sky pixel. Sky pixels can not be declared as shadow pixels.


> "settings.general.openCL.platformid" specifies platform index used in openCL processing

> "settings.general.openCL.deviceid" specifies device index of selected platform used in openCL processing

> "settings.general.openCL.UsePrecompiledKernels" flag which indicates if use precompiled OpenCL kernel binaries (_Notice: currently affects only GPUs devices_)

> "settings.general.openMP.threadNum" determine number of threads used by OpenMP in non OpenCL builds

> "settings.general.Training.svm.svm\_type" specifies svm type used in creating model file (_see comment for more_)

> "settings.general.Training.svm.kernel\_type" specifies kernel type used in creating model file (_see comment for more_)

# Other Varibles #
> For all other properties variables please look at comments in configuration file