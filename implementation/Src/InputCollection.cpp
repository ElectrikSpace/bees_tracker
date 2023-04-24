//
// Created by sylvain on 31/01/23.
//

#include "InputCollection.hpp"

#include <cmath>

#define INPUT_COUNT 3

static std::string inputNames[INPUT_COUNT] = {"video1",
                                              "video2",
                                              "video3"};

static std::string framePaths[INPUT_COUNT] = {"../data/video1_frames/",
                                              "../data/video2_frames/",
                                              "../data/video3_frames/"};

static FrameResolution_t frameResolutions[INPUT_COUNT] = {{1080, 1920},
                                                          {480, 1920},
                                                          {1080, 920}};

static uint32_t frameBaseRow[INPUT_COUNT] = {0, 600, 0};
static uint32_t frameBaseCol[INPUT_COUNT] = {0, 0, 1000};

static std::size_t frameCount [INPUT_COUNT] = {1675, 1347, 3095};

static uint32_t entranceRectangleCount[INPUT_COUNT] = {1, 1, 1};
static uint32_t bordersRectangleCount[INPUT_COUNT] = {3, 3, 3};

static Rectangle_t input1Entrance[1] = {/* TODO */};
static Rectangle_t input2Entrance[1] = {{{0, 75}, {90, 1750}}};
static Rectangle_t input3Entrance[1] = {/* TODO */};

static Rectangle_t input1Borders[3] = {/* TODO */};
static Rectangle_t input2Borders[3] = {{{0, 0}, {480, 100}},
                                       {{480-100, 0}, {480, 1920}},
                                       {{0, 1920-100}, {480, 1920}}};
static Rectangle_t input3Borders[3] = {/* TODO */};

std::size_t InputCollection::getInputsCount() {
    return INPUT_COUNT;
}

std::size_t InputCollection::getInputFrameCount(uint32_t inputIndex) {
    return frameCount[inputIndex];
}

std::shared_ptr<Input_t> InputCollection::createInput(uint32_t inputIndex, uint32_t resolutionDivisor) {
    auto input = std::make_shared<Input_t>();

    input->inputIndex = inputIndex;
    input->name = inputNames[inputIndex];
    input->framesPath = framePaths[inputIndex];
    input->baseRow = frameBaseRow[inputIndex];
    input->baseCol = frameBaseCol[inputIndex];
    input->resolutionDivider = resolutionDivisor;
    input->inputResolution = frameResolutions[inputIndex];
    if (resolutionDivisor != 1) {
        float oneDimensionDivider = std::sqrt((float) resolutionDivisor);
        input->usedResolution.rows = (std::size_t) ((float) frameResolutions[inputIndex].rows / oneDimensionDivider);
        input->usedResolution.cols = (std::size_t) ((float) frameResolutions[inputIndex].cols / oneDimensionDivider);
    }
    else {
        input->usedResolution.rows = frameResolutions[inputIndex].rows;
        input->usedResolution.cols = frameResolutions[inputIndex].cols;
    }

    Rectangle_t* entranceRectPtr;
    Rectangle_t* bordersRectPtr;
    switch (inputIndex) {
        case 0:
            entranceRectPtr = input1Entrance;
            bordersRectPtr = input1Borders;
            break;
        case 1:
            entranceRectPtr = input2Entrance;
            bordersRectPtr = input2Borders;
            break;
        case 2:
        default:
            entranceRectPtr = input3Entrance;
            bordersRectPtr = input3Borders;
    }

    // set entrance rectangles
    input->entranceRectangles.clear();
    for (uint32_t i = 0; i < entranceRectangleCount[inputIndex]; i++)
        input->entranceRectangles.push_back(entranceRectPtr[i]);

    // set border rectangles
    input->borderRectangles.clear();
    for (uint32_t i = 0; i < bordersRectangleCount[inputIndex]; i++)
        input->borderRectangles.push_back(bordersRectPtr[i]);

    return std::move(input);
}
