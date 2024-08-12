#include <iostream>

#include "opencv2/opencv.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

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

int main(int argc, char **argv){
  //VideoCapture Object
  VideoCapture cap(argv[1]); 
    
  // Check if capture opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  Mat frame, roi, hsv_roi, mask;

  //Initial Rectangle
  Rect track_window(800, 400, 75, 75);

  while(1){
    //Store Frame
    cap >> frame;

    //If the frame is empty, break immediately
    if (frame.empty())
      break;
 
    TrackBall(frame, roi, hsv_roi, mask, track_window);

    imshow("Frame", frame);

    //Press ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
  }
  
  // When everything done, release the video capture object
  cap.release();
 
  // Closes all the frames
  destroyAllWindows();
     
  return 0;
}