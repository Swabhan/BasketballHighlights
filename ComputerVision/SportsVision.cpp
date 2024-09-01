/**
 * @file SportsVision.cpp
 * @author Swabhan Katkoori
 * 
 * Header File, initialize Sports Vision Class
 */

#pragma once

#include "SportsVision.h"

SportsVision::SportsVision()
{
    model_path = "/Users/swabhankatkoori/Documents/Development/BasketballHighlights/ComputerVision/yolov7-tiny.onnx";
    image_path = "/Users/swabhankatkoori/Documents/Development/BasketballHighlights/yolov7/inference/images/image2.jpg";


}

void SportsVision::SetFrameRate(int frame){
    frameRate = frame;
};

int SportsVision::GetFrameRate(){
    return frameRate;
}

const std::array<const char*, 80> SportsVision::class_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
};