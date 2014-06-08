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
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU-Linux-x86
CND_DLIB_EXT=so
CND_CONF=ReleaseOpenCL_AMD
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/main.o \
	${OBJECTDIR}/src/shadowdetection/opencl/OpenCLTools.o \
	${OBJECTDIR}/src/shadowdetection/opencv/OpenCV2Tools.o \
	${OBJECTDIR}/src/shadowdetection/opencv/OpenCVTools.o \
	${OBJECTDIR}/src/shadowdetection/util/Cofig.o \
	${OBJECTDIR}/src/shadowdetection/util/TabParser.o


# C Compiler Flags
CFLAGS=-m64

# CC Compiler Flags
CCFLAGS=-m64
CXXFLAGS=-m64

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-L/usr/local/lib -L/opt/AMDAPP/lib/x86_64

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/shadowdetection

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/shadowdetection: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/shadowdetection ${OBJECTFILES} ${LDLIBSOPTIONS} -lOpenCL -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_ocl

${OBJECTDIR}/main.o: main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -D_AMD -D_OPENCL -I/usr/local/include/opencv -I/opt/AMDAPP/include -Isrc -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.cpp

${OBJECTDIR}/src/shadowdetection/opencl/OpenCLTools.o: src/shadowdetection/opencl/OpenCLTools.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/shadowdetection/opencl
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -D_AMD -D_OPENCL -I/usr/local/include/opencv -I/opt/AMDAPP/include -Isrc -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/shadowdetection/opencl/OpenCLTools.o src/shadowdetection/opencl/OpenCLTools.cpp

${OBJECTDIR}/src/shadowdetection/opencv/OpenCV2Tools.o: src/shadowdetection/opencv/OpenCV2Tools.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/shadowdetection/opencv
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -D_AMD -D_OPENCL -I/usr/local/include/opencv -I/opt/AMDAPP/include -Isrc -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/shadowdetection/opencv/OpenCV2Tools.o src/shadowdetection/opencv/OpenCV2Tools.cpp

${OBJECTDIR}/src/shadowdetection/opencv/OpenCVTools.o: src/shadowdetection/opencv/OpenCVTools.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/shadowdetection/opencv
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -D_AMD -D_OPENCL -I/usr/local/include/opencv -I/opt/AMDAPP/include -Isrc -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/shadowdetection/opencv/OpenCVTools.o src/shadowdetection/opencv/OpenCVTools.cpp

${OBJECTDIR}/src/shadowdetection/util/Cofig.o: src/shadowdetection/util/Cofig.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/shadowdetection/util
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -D_AMD -D_OPENCL -I/usr/local/include/opencv -I/opt/AMDAPP/include -Isrc -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/shadowdetection/util/Cofig.o src/shadowdetection/util/Cofig.cpp

${OBJECTDIR}/src/shadowdetection/util/TabParser.o: src/shadowdetection/util/TabParser.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/shadowdetection/util
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -D_AMD -D_OPENCL -I/usr/local/include/opencv -I/opt/AMDAPP/include -Isrc -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/shadowdetection/util/TabParser.o src/shadowdetection/util/TabParser.cpp

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
