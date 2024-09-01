cmake_minimum_required(VERSION 3.12)

# Project name
project(ComputerVision)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find and include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Manually specify ONNX Runtime paths
set(ONNX_RUNTIME_INCLUDE_DIR /usr/local/include)
set(ONNX_RUNTIME_LIB_DIR /usr/local/lib)

# Include ONNX Runtime headers
include_directories(${ONNX_RUNTIME_INCLUDE_DIR})

# Specify source files
set(SOURCE_FILES main.cpp pch.h SportsVision.h SportsVision.cpp BodyPoseComp.h BodyPoseComp.cpp)

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Link the executable with OpenCV and ONNX Runtime libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNX_RUNTIME_LIB_DIR}/libonnxruntime.dylib)

# Specify the directory for ONNX Runtime library
link_directories(${ONNX_RUNTIME_LIB_DIR})
