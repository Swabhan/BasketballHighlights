#include <iostream>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "opencv2/opencv.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/tracking.hpp>



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

bool TrackBall(Mat &frame, Mat &roi, Mat &hsv_roi, Mat &mask, Rect &track_window)
{
  Mat hsv, dst;
  
  // Setup the termination criteria, either 10 iteration or move by at least 1 pt
  TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 100, 1);

  int image_width = frame.cols;
  int image_height = frame.rows;

  roi = frame(track_window);
  cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
  inRange(hsv_roi, Scalar(0, 60, 32), Scalar(180, 255, 255), mask);

  float range_[] = {0, 180};
  const float* range[] = {range_};
  Mat roi_hist;
  int histSize[] = {180};
  int channels[] = {0};
  calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
  normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

  cvtColor(frame, hsv, COLOR_BGR2HSV);
  calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

  // apply meanshift to get the new location
  meanShift(dst, track_window, term_crit);

  // Draw it on image
  rectangle(frame, track_window, 255, 2);

  return true;
}

std::tuple<Array, Shape, cv::Mat> read_image(cv::Mat frame, int size)
{
    auto image = frame;
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

Rect display_image(cv::Mat image, const Array &output, const Shape &shape)
{
    for (size_t i = 0; i < shape[0]; ++i)
    {
        auto ptr = output.data() + i * shape[1];

        int x = ptr[1], y = ptr[2], w = ptr[3] - x, h = ptr[4] - y, c = ptr[5];

        auto color = CV_RGB(255, 255, 255);
        auto name = std::string(class_names[c]) + ":" + std::to_string(int(ptr[6] * 100)) + "%";
        cv::rectangle(image, {x, y, w, h}, color);
        cv::putText(image, name, {x, y}, cv::FONT_HERSHEY_DUPLEX, 1, color);

        if(class_names[c] == "frisbee"){

          imshow("Frame", image);

          cv::waitKey(0);

          Rect track_window(x, y, w, h);
          return track_window;
        }
    }

    imshow("Frame", image);

    cv::waitKey(0);



    Rect track_window(800, 400, 75, 75);
    return track_window;
}

int main(int argc, char **argv){
  VideoCapture cap(argv[1]); 

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

  cap >> frame;
  
  auto [array, shape, image] = read_image(frame, 640);
  auto [output, output_shape] = process_image(session, array, shape);
  Rect track_window = display_image(image, output, output_shape);

  while(0){
    //Store Frame
    cap >> frame;

    //If the frame is empty, break immediately
    if (frame.empty())
      break;
 
    TrackBall(frame, roi, hsv_roi, mask, track_window);
    imshow("Frame", frame);

    cv::waitKey(0);


    //Press ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      session.release();
      break;
  }
  
  // When everything done, release the video capture object
  cap.release();
 
  // Closes all the frames
  destroyAllWindows();
     
  return 0;
}