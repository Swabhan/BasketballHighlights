/**
 * @file pch.h
 * @author Swabhan Katkoori
 * 
 * Pre-compiled Header File
 */

#ifndef PCH_H
#define PCH_H

// Standard library headers
#include <iostream>
#include <vector>
#include <cassert>
#include <tuple>

// ONNX Runtime headers
#include <onnxruntime/onnxruntime_cxx_api.h>

// OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/tracking.hpp>

// Namespace usage
using namespace std;
using namespace cv;
using namespace cv::dnn;

#endif // PCH_H
