/**
 * @file BodyPoseComp.h
 * @author Swabhan Katkoori
 * 
 * Header File, initialize Body Pose Comparision
 */

#pragma once

#include "pch.h"
#include "SportsVision.h"

class BodyPoseComp : public SportsVision {
    private: 
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

        Net net = cv::dnn::readNetFromCaffe(protoFile, weightsFile);

        // Specify the input image dimensions
        int inWidth = 368;
        int inHeight = 368;
        float thresh = 0.1;  

        int frameCount = 0;

    public:
        BodyPoseComp();

        void ApplyPoseStructure(Mat frame);
        void PrepareImage(Mat frame);
        void ApplyHumanPoseEst(Mat frame);


};