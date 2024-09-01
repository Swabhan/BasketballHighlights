#pragma once

#include "BodyPoseComp.h"

BodyPoseComp::BodyPoseComp(){

}

void BodyPoseComp::ApplyPoseStructure(Mat frame){
    if(frameCount % GetFrameRate() == 0){
      PrepareImage(frame);
      ApplyHumanPoseEst(frame);
    }

    imshow("Frame", frame);
    frameCount = frameCount + 1;

}

void BodyPoseComp::PrepareImage(Mat frame){
    // Prepare the frame to be fed to the network
    Mat inpBlob = cv::dnn::blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);
    
    // Set the prepared object as the input blob of the network
    net.setInput(inpBlob);
};

void BodyPoseComp::ApplyHumanPoseEst(Mat frame){
    int frameWidth = frame.cols;
    int frameHeight = frame.rows;

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
    
        line(frame, partA, partB, Scalar(0,255,255), 4);
        circle(frame, partA, 8, Scalar(0,0,255), -1);
        circle(frame, partB, 8, Scalar(0,0,255), -1);
    }

    
}