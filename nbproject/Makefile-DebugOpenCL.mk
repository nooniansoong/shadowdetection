#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=nvcc
CCC=nvcc
CXX=nvcc
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU+CUDA-Linux-x86
CND_DLIB_EXT=so
CND_CONF=DebugOpenCL
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/main.o \
	${OBJECTDIR}/src/cpp/core/opencl/OpenClToolsBase.o \
	${OBJECTDIR}/src/cpp/core/opencl/libsvm/OpenCLToolsPredict.o \
	${OBJECTDIR}/src/cpp/core/opencl/libsvm/OpenCLToolsTrain.o \
	${OBJECTDIR}/src/cpp/core/opencv/OpenCV2Tools.o \
	${OBJECTDIR}/src/cpp/core/opencv/OpenCVTools.o \
	${OBJECTDIR}/src/cpp/core/tools/svm/TrainingSet.o \
	${OBJECTDIR}/src/cpp/core/tools/svm/libsvmopenmp/svm-train.o \
	${OBJECTDIR}/src/cpp/core/util/Cofig.o \
	${OBJECTDIR}/src/cpp/core/util/MemMenager.o \
	${OBJECTDIR}/src/cpp/core/util/ParametersFactory.o \
	${OBJECTDIR}/src/cpp/core/util/PredictorFactory.o \
	${OBJECTDIR}/src/cpp/core/util/TabParser.o \
	${OBJECTDIR}/src/cpp/core/util/Timer.o \
	${OBJECTDIR}/src/cpp/core/util/predicition/libsvm/SvmPredict.o \
	${OBJECTDIR}/src/cpp/core/util/predicition/regression/RegressionPredict.o \
	${OBJECTDIR}/src/cpp/core/util/rtti/RTTI.o \
	${OBJECTDIR}/src/cpp/shadowdetection/opencl/OpenCLImageParameters.o \
	${OBJECTDIR}/src/cpp/shadowdetection/opencl/OpenCLTools.o \
	${OBJECTDIR}/src/cpp/shadowdetection/opencl/OpenCLToolsImage.o \
	${OBJECTDIR}/src/cpp/shadowdetection/tools/image/ImageShadowParameters.o \
	${OBJECTDIR}/src/cpp/shadowdetection/tools/image/ResultFixer.o \
	${OBJECTDIR}/src/cpp/skydetection/SkyDetection.o \
	${OBJECTDIR}/src/cpp/thirdparty/lib_svm/svm.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=-m64
CXXFLAGS=-m64

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-L/usr/local/lib

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/shadowdetection

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/shadowdetection: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/shadowdetection ${OBJECTFILES} ${LDLIBSOPTIONS} -lOpenCL -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_ocl

${OBJECTDIR}/main.o: main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/main.o main.cpp

${OBJECTDIR}/src/cpp/core/opencl/OpenClToolsBase.o: src/cpp/core/opencl/OpenClToolsBase.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/opencl
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/opencl/OpenClToolsBase.o src/cpp/core/opencl/OpenClToolsBase.cpp

${OBJECTDIR}/src/cpp/core/opencl/libsvm/OpenCLToolsPredict.o: src/cpp/core/opencl/libsvm/OpenCLToolsPredict.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/opencl/libsvm
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/opencl/libsvm/OpenCLToolsPredict.o src/cpp/core/opencl/libsvm/OpenCLToolsPredict.cpp

${OBJECTDIR}/src/cpp/core/opencl/libsvm/OpenCLToolsTrain.o: src/cpp/core/opencl/libsvm/OpenCLToolsTrain.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/opencl/libsvm
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/opencl/libsvm/OpenCLToolsTrain.o src/cpp/core/opencl/libsvm/OpenCLToolsTrain.cpp

${OBJECTDIR}/src/cpp/core/opencv/OpenCV2Tools.o: src/cpp/core/opencv/OpenCV2Tools.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/opencv
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/opencv/OpenCV2Tools.o src/cpp/core/opencv/OpenCV2Tools.cpp

${OBJECTDIR}/src/cpp/core/opencv/OpenCVTools.o: src/cpp/core/opencv/OpenCVTools.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/opencv
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/opencv/OpenCVTools.o src/cpp/core/opencv/OpenCVTools.cpp

${OBJECTDIR}/src/cpp/core/tools/svm/TrainingSet.o: src/cpp/core/tools/svm/TrainingSet.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/tools/svm
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/tools/svm/TrainingSet.o src/cpp/core/tools/svm/TrainingSet.cpp

${OBJECTDIR}/src/cpp/core/tools/svm/libsvmopenmp/svm-train.o: src/cpp/core/tools/svm/libsvmopenmp/svm-train.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/tools/svm/libsvmopenmp
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/tools/svm/libsvmopenmp/svm-train.o src/cpp/core/tools/svm/libsvmopenmp/svm-train.cpp

${OBJECTDIR}/src/cpp/core/util/Cofig.o: src/cpp/core/util/Cofig.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/util
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/util/Cofig.o src/cpp/core/util/Cofig.cpp

${OBJECTDIR}/src/cpp/core/util/MemMenager.o: src/cpp/core/util/MemMenager.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/util
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/util/MemMenager.o src/cpp/core/util/MemMenager.cpp

${OBJECTDIR}/src/cpp/core/util/ParametersFactory.o: src/cpp/core/util/ParametersFactory.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/util
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/util/ParametersFactory.o src/cpp/core/util/ParametersFactory.cpp

${OBJECTDIR}/src/cpp/core/util/PredictorFactory.o: src/cpp/core/util/PredictorFactory.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/util
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/util/PredictorFactory.o src/cpp/core/util/PredictorFactory.cpp

${OBJECTDIR}/src/cpp/core/util/TabParser.o: src/cpp/core/util/TabParser.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/util
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/util/TabParser.o src/cpp/core/util/TabParser.cpp

${OBJECTDIR}/src/cpp/core/util/Timer.o: src/cpp/core/util/Timer.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/util
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/util/Timer.o src/cpp/core/util/Timer.cpp

${OBJECTDIR}/src/cpp/core/util/predicition/libsvm/SvmPredict.o: src/cpp/core/util/predicition/libsvm/SvmPredict.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/util/predicition/libsvm
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/util/predicition/libsvm/SvmPredict.o src/cpp/core/util/predicition/libsvm/SvmPredict.cpp

${OBJECTDIR}/src/cpp/core/util/predicition/regression/RegressionPredict.o: src/cpp/core/util/predicition/regression/RegressionPredict.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/util/predicition/regression
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/util/predicition/regression/RegressionPredict.o src/cpp/core/util/predicition/regression/RegressionPredict.cpp

${OBJECTDIR}/src/cpp/core/util/rtti/RTTI.o: src/cpp/core/util/rtti/RTTI.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/core/util/rtti
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/core/util/rtti/RTTI.o src/cpp/core/util/rtti/RTTI.cpp

${OBJECTDIR}/src/cpp/shadowdetection/opencl/OpenCLImageParameters.o: src/cpp/shadowdetection/opencl/OpenCLImageParameters.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/shadowdetection/opencl
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/shadowdetection/opencl/OpenCLImageParameters.o src/cpp/shadowdetection/opencl/OpenCLImageParameters.cpp

${OBJECTDIR}/src/cpp/shadowdetection/opencl/OpenCLTools.o: src/cpp/shadowdetection/opencl/OpenCLTools.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/shadowdetection/opencl
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/shadowdetection/opencl/OpenCLTools.o src/cpp/shadowdetection/opencl/OpenCLTools.cpp

${OBJECTDIR}/src/cpp/shadowdetection/opencl/OpenCLToolsImage.o: src/cpp/shadowdetection/opencl/OpenCLToolsImage.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/shadowdetection/opencl
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/shadowdetection/opencl/OpenCLToolsImage.o src/cpp/shadowdetection/opencl/OpenCLToolsImage.cpp

${OBJECTDIR}/src/cpp/shadowdetection/tools/image/ImageShadowParameters.o: src/cpp/shadowdetection/tools/image/ImageShadowParameters.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/shadowdetection/tools/image
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/shadowdetection/tools/image/ImageShadowParameters.o src/cpp/shadowdetection/tools/image/ImageShadowParameters.cpp

${OBJECTDIR}/src/cpp/shadowdetection/tools/image/ResultFixer.o: src/cpp/shadowdetection/tools/image/ResultFixer.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/shadowdetection/tools/image
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/shadowdetection/tools/image/ResultFixer.o src/cpp/shadowdetection/tools/image/ResultFixer.cpp

${OBJECTDIR}/src/cpp/skydetection/SkyDetection.o: src/cpp/skydetection/SkyDetection.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/skydetection
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/skydetection/SkyDetection.o src/cpp/skydetection/SkyDetection.cpp

${OBJECTDIR}/src/cpp/thirdparty/lib_svm/svm.o: src/cpp/thirdparty/lib_svm/svm.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cpp/thirdparty/lib_svm
	${RM} "$@.d"
	$(COMPILE.cc) -g -D_OPENCL -I/usr/local/include/opencv -Isrc/cpp -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cpp/thirdparty/lib_svm/svm.o src/cpp/thirdparty/lib_svm/svm.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/shadowdetection

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
