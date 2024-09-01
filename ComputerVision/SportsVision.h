/**
 * @file SportsVision.h
 * @author Swabhan Katkoori
 * 
 * Header File, initialize Sports Vision Class
 */

#pragma once

#include "pch.h"

class SportsVision{
    private:
        int frameRate = 0;

        using Array = std::vector<float>;
        using Shape = std::vector<int64_t>;

        bool use_cuda = false;
        int image_size = 640;

        std::string model_path;
        std::string image_path;

        static const std::array<const char*, 80> class_names;

        float heightScale = 1;
        float widthScale = 1;


    public:
        SportsVision();

        //Set Frame Rate for computer vision interaction
        void SetFrameRate(int frame);
        int GetFrameRate();

};
