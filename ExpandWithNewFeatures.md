# Use other features #

> If you need different set of features than current set you can put in process easily in only three steps:
  * Implement new class derived from class IImageParameteres (defined in src/cpp/core/tools/image/IImageParameteres.h)
  * In function createImageParameters (defined in src/cpp/util/ParametersFactory.h implemented in ParametersFactory.cpp) change return type to return your class. Something like:
```
        IImageParameteres* createImageParameters(){
            return new YourNewParameterClass();
        }
```
  * Rebuild project configuration you use

> I know that this is not good solution, so soon it will be enabled some simple RTTI which will provide enabling of new class through configuration file.

# Explanation of IImageParameteres interface #
> ## Method ##
```
virtual core::util::Matrix<float>* getImageParameters(const cv::Mat& originalImage,
                                                      const cv::Mat& hsvImage, 
                                                      const cv::Mat& hlsImage, 
                                                      int& rowDimension, 
                                                      int& pixelNum) throw (SDException&) = 0;
```
> > ### Usage ###
> > > This method is used in prediction process.

> > ### Input parameters ###
> > > Input parameters are:
        * originalImage represents original image in BGR format.
        * hsvImage represents original image converted to HSV color space
        * hlsImage represents original image converted to HLS color space

> > ### Output parameters ###
> > > Output parameters are:
        * rowDimension actually represents number of features per pixel
        * pixelNum represents pixel number in image

> > ### Returns ###
> > > Pointer to matrix class instance object which represents all features per pixel. First dimension is pixel index, second is feature index.

> > ### Notices ###
      * Don't forget to release return value after.

> ## Method ##
```
  virtual core::util::Matrix<float>* getImageParameters(const cv::Mat& originalImage, 
                                                        const cv::Mat& maskImage,                                                     
                                                        int& rowDimension, 
                                                        int& pixelNum) throw (SDException&) = 0;
```
> > ### Usage ###
> > > This method is used in training process

> > ### Input parameters ###
> > > Input parameters are:
        * originalImage represents original image in BGR format.
        * maskImage is single channel image which represents results of detection

> > ### Output parameters ###
> > > Output parameters are:
        * rowDimension actually represents number of features per pixel
        * pixelNum represents pixel number in image

> > ### Returns ###
> > > Pointer to matrix class instance object which represents detection result for pixel + all features per pixel. First dimension is pixel index, second zero index is prediction result and after are features.

> > ### Notices ###
      * Best is that this method calls previous method for acquiring pixel features
      * Don't forget to release return value after.