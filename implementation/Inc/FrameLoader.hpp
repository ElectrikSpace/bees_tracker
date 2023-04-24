//
// Created by sylvain on 28/01/23.
//

#ifndef EXERCISE5_FRAME_LOADER_HPP
#define EXERCISE5_FRAME_LOADER_HPP

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "Parameters.hpp"

#include <exception>
#include <iostream>

class FrameLoader {
private:
    std::vector<std::unique_ptr<cv::Mat>> initializationFrames;
    std::vector<std::unique_ptr<cv::Mat>> frames;
    FrameResolution_t framesResolution{};
    std::size_t frameCount;

public:
    FrameLoader(Input_t& input, std::size_t frameCount)
    {
        this->frameCount = frameCount;
        framesResolution = input.usedResolution;

        std::vector<std::unique_ptr<cv::Mat>>* list = &initializationFrames;
        for (std::size_t i = 0; i < (frameCount + Parameters::STORED_FRAMES); i++) {
            std::size_t endX = input.baseRow + input.inputResolution.rows;
            std::size_t endY = input.baseCol + input.inputResolution.cols;
            auto rawFrame = std::make_unique<cv::Mat>(cv::imread(input.framesPath + "frame" + std::to_string(i) + ".jpg")(
                cv::Range((int) input.baseRow, (int) endX),
                cv::Range((int) input.baseCol, (int) endY)));

            if (i >= Parameters::STORED_FRAMES)
                list = &frames;

            cv::Mat frame;
            cv::resize(*rawFrame, frame, cv::Size((int) framesResolution.cols, (int) framesResolution.rows));
            list->push_back(std::move(std::make_unique<cv::Mat>(frame)));
        }
    }

    cv::Mat& getFrameMat(std::size_t index)
    {
        if (index < frames.size())
            return *frames[index];

        throw std::logic_error("index out of range");
    }

    uint8_t* getFramePtr(std::size_t index)
    {
        if (index < frames.size())
            return frames[index]->data;

        throw std::logic_error("index out of range");
    }

    cv::Mat& getInitializationFrameMat(std::size_t index)
    {
        if (index < initializationFrames.size())
            return *initializationFrames[index];

        throw std::logic_error("index out of range");
    }

    uint8_t* getInitializationFramePtr(std::size_t index)
    {
        if (index < initializationFrames.size())
            return initializationFrames[index]->data;

        throw std::logic_error("index out of range");
    }

    FrameResolution_t getFramesResolution()
    {
        return framesResolution;
    }

    std::size_t getFrameCount()
    {
        return frameCount;
    }
};


#endif //EXERCISE5_FRAME_LOADER_HPP
