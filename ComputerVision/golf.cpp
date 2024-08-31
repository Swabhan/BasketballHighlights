#include <iostream>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "opencv2/opencv.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/ocl.hpp>
#include "opencv2/imgproc/types_c.h"
#include <opencv2/tracking/tracking.hpp>


#include <vector>
#include <cassert>
#include <tuple>

using namespace std;
using namespace cv;

//Onnx + Yolo
using Array = std::vector<float>;
using Shape = std::vector<int64_t>;

bool use_cuda = false;
int image_size = 640;
std::string model_path = "/Users/swabhankatkoori/Documents/Development/BasketballHighlights/ComputerVision/yolov7-tiny.onnx";
std::string image_path = "/Users/swabhankatkoori/Documents/Development/BasketballHighlights/yolov7/inference/images/image2.jpg";

const char *class_names[] = {
  "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
  "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
  "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
  "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
  "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
  "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
  "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
  "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
  "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
  "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
  "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
  "teddy bear",     "hair drier", "toothbrush"};

float heightScale = 1;
float widthScale = 1;

std::tuple<Array, Shape, cv::Mat> read_image(cv::Mat frame, int size)
{
    auto image = frame;

    heightScale = float(image.size().height) / float(size);
    widthScale = float(image.size().width) / float(size);

    assert(!image.empty() && image.channels() == 3);

    cv::resize(image, image, {size, size});

    Shape shape = {1, image.channels(), image.rows, image.cols};
    cv::Mat nchw = cv::dnn::blobFromImage(image, 1.0, {}, {}, true) / 255.f;

    Array array(nchw.ptr<float>(), nchw.ptr<float>() + nchw.total());
    return {array, shape, image};
}

std::pair<Array, Shape> process_image(Ort::Session &session, Array &array, Shape shape)
{ 
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    auto input = Ort::Value::CreateTensor<float>(
        memory_info, (float *)array.data(), array.size(), shape.data(), shape.size());


    const char *input_names[] = {"images"};
    const char *output_names[] = {"output"};

    auto output = session.Run({}, input_names, &input, 1, output_names, 1);
    shape = output[0].GetTensorTypeAndShapeInfo().GetShape();

    auto ptr = output[0].GetTensorData<float>();

    return {Array(ptr, ptr + shape[0] * shape[1]), shape};
}

Rect2d display_image(cv::Mat image, const Array &output, const Shape &shape)
{
    for (size_t i = 0; i < shape[0]; ++i)
    {
        auto ptr = output.data() + i * shape[1];

        int x = int(ptr[1] * widthScale), y = int(ptr[2] * heightScale), w = int(ptr[3] * widthScale - x), h = int(ptr[4] * heightScale - y ), c = ptr[5];

        auto color = CV_RGB(255, 255, 255);
        auto name = std::string(class_names[c]) + ":" + std::to_string(int(ptr[6] * 100)) + "%";
        cv::rectangle(image, {x, y, w, h}, color);
        cv::putText(image, name, {x, y}, cv::FONT_HERSHEY_DUPLEX, 1, color);

         if(class_names[c] == "sports ball"){
          Rect2d bbox(x, y, w, h);
          return bbox;
        }
    }

    Rect2d bbox(568, 454, 92, 79);
    return bbox;
}

int main(int argc, char **argv){
  //Open Video
  VideoCapture cap(argv[1]); 

  //Open ONNX Session
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv7");
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  Ort::Session session(env, model_path.c_str(), options);
    
  // Check if capture opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  Mat frame, roi, hsv_roi, mask;

  //Initialize tracking variables
  Ptr<Tracker> tracker = TrackerMIL::create();
  int frameCount = 0;

  cap >> frame;
  
  cv::Rect2d bbox(730, 810, 40, 30);
  
//   auto [array, shape, image] = read_image(frame, 640);
//   auto [output, output_shape] = process_image(session, array, shape);
//   bbox = display_image(image, output, output_shape);

  Rect bbox_int = Rect(static_cast<int>(bbox.x), static_cast<int>(bbox.y),
                  static_cast<int>(bbox.width), static_cast<int>(bbox.height));

  tracker->init(frame, bbox);

  bool tracking = true;
  bool ok = true;

  while(1){
    //Store Frame
    cap >> frame;

    //Recalibrate Object Location
    // if(frameCount % 120 == 0 || (tracking == false && frameCount % 60)){
    //   auto [array, shape, image] = read_image(frame, 640);
    //   auto [output, output_shape] = process_image(session, array, shape);
    //   bbox = display_image(image, output, output_shape);
    //   bbox_int = Rect(static_cast<int>(bbox.x), static_cast<int>(bbox.y),
    //             static_cast<int>(bbox.width), static_cast<int>(bbox.height));

      
    //   if(bbox_int.x != 568){
    //     tracker->init(frame, bbox);
    //     tracking = true;
    //   }

    // }

    if(tracking == true){
      ok = tracker->update(frame, bbox_int);
    } else{
      ok = false;
      tracking = false;
    }
   

    if (ok) {
      cv::rectangle(frame, bbox_int, cv::Scalar(255, 0, 0), 2, 1);
    }

    // //If the frame is empty, break immediately
    if (frame.empty())
      break;

    //Show frame
    imshow("Frame", frame);


    frameCount = frameCount + 1;

    //Press ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      // session.release();
      break;
  }
  session.release();

  // When everything done, release the video capture object
  cap.release();
 
  // Closes all the frames
  destroyAllWindows();
     
  return 0;
}
