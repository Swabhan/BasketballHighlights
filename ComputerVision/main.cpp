#include <iostream>


#include <onnxruntime/onnxruntime_cxx_api.h>

#include "opencv2/opencv.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include "opencv2/imgproc/types_c.h"
#include <opencv2/tracking/tracking.hpp>

#include <vector>
#include <cassert>
#include <tuple>

using namespace std;
using namespace cv;
using namespace cv::dnn;

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

int trackingInit(VideoCapture &cap){
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
  Ptr<Tracker> tracker = TrackerCSRT::create();
  int frameCount = 0;

  cap >> frame;
  
  cv::Rect2d bbox(568, 454, 92, 79);
  
  auto [array, shape, image] = read_image(frame, 640);
  auto [output, output_shape] = process_image(session, array, shape);
  bbox = display_image(image, output, output_shape);
  Rect bbox_int = Rect(static_cast<int>(bbox.x), static_cast<int>(bbox.y),
                  static_cast<int>(bbox.width), static_cast<int>(bbox.height));

  tracker->init(frame, bbox);

  bool tracking = true;
  bool ok = true;

  while(1){
    //Store Frame
    cap >> frame;

    //Recalibrate Object Location
    if(frameCount % 120 == 0 || (tracking == false && frameCount % 60)){
      auto [array, shape, image] = read_image(frame, 640);
      auto [output, output_shape] = process_image(session, array, shape);
      bbox = display_image(image, output, output_shape);
      bbox_int = Rect(static_cast<int>(bbox.x), static_cast<int>(bbox.y),
                static_cast<int>(bbox.width), static_cast<int>(bbox.height));

      
      if(bbox_int.x != 568){
        tracker->init(frame, bbox);
        tracking = true;
      }

    }

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

  return 0;
}


const int POSE_PAIRS[14][2] = 
  {   
      {0,1}, {1,2}, {2,3},
      {3,4}, {1,5}, {5,6},
      {6,7}, {1,14}, {14,8}, {8,9},
      {9,10}, {14,11}, {11,12}, {12,13}
  };

string protoFile = "/Users/swabhankatkoori/Documents/Development/BasketballHighlights/openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
string weightsFile = "/Users/swabhankatkoori/Documents/Development/BasketballHighlights/openpose/models/pose/mpi/pose_iter_160000.caffemodel";

int nPoints = 15;
string device = "cpu";

int main(int argc, char **argv){
  //Open Video
  VideoCapture cap(argv[1]); 
  //trackingInit(cap);

  Net net = cv::dnn::readNetFromCaffe(protoFile, weightsFile);
  Mat frame = imread("cade.jpg");

  if (frame.empty()) {
    std::cerr << "Error: Unable to load image!" << std::endl;
    return -1;
  }

  while(1){
    //Store Frame
    cap >> frame;

    // Specify the input image dimensions
    int inWidth = 368;
    int inHeight = 368;
    float thresh = 0.1;  

    int frameWidth = frame.cols;
    int frameHeight = frame.rows;
    
    // Prepare the frame to be fed to the network
    Mat inpBlob = cv::dnn::blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);
    
    // Set the prepared object as the input blob of the network
    net.setInput(inpBlob);


    Mat output = net.forward();

    int H = output.size[2];
    int W = output.size[3];

    net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    
    // find the position of the body parts
    vector<Point> points(nPoints);
    for (int n=0; n < nPoints; n++)
    {
      // Probability map of corresponding body's part.
      Mat probMap(H, W, CV_32F, output.ptr(0,n));

      Point2f p(-1,-1);
      Point maxLoc;
      double prob;
      minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
      if (prob > thresh)
      {
          p = maxLoc;
          p.x *= (float)frameWidth / W ;
          p.y *= (float)frameHeight / H ;

          circle(frame, cv::Point((int)p.x, (int)p.y), 8, Scalar(0,255,255), -1);
          cv::putText(frame, cv::format("%d", n), cv::Point((int)p.x, (int)p.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

      }
      points[n] = p;
    }


    int nPairs = sizeof(POSE_PAIRS)/sizeof(POSE_PAIRS[0]);
    
    for (int n = 0; n < nPairs; n++)
    {
        // lookup 2 connected body/hand parts
        Point2f partA = points[POSE_PAIRS[n][0]];
        Point2f partB = points[POSE_PAIRS[n][1]];
    
        if (partA.x<=0 || partA.y<=0 || partB.x<=0 || partB.y<=0)
            continue;
    
        line(frame, partA, partB, Scalar(0,255,255), 8);
        circle(frame, partA, 8, Scalar(0,0,255), -1);
        circle(frame, partB, 8, Scalar(0,0,255), -1);
    }

    imshow("Frame", frame);

    char c=(char)waitKey(25);
    if(c==27)
      // session.release();
      break;
  }

  // Closes all the frames
  destroyAllWindows();
     
  return 0;
}