//
// Created by sylvain on 29/01/23.
//

#ifndef EXERCISE5_STRUCTURES_HPP
#define EXERCISE5_STRUCTURES_HPP

#include <cstdint>

typedef struct{
    std::size_t rows;
    std::size_t cols;
} FrameResolution_t;

typedef struct{
    uint32_t x;
    uint32_t y;
} Point_t;

typedef struct{
    float x;
    float y;
} FloatPoint_t;

typedef struct{
    Point_t p1;
    Point_t p2;
} Rectangle_t;

typedef struct{
    int16_t r;
    int16_t g;
    int16_t b;
} Pixel_t;

typedef struct{
    int16_t thMin;
    int16_t thMax;
    int16_t thFactor;
} Threshold1_t;

typedef struct{
    uint16_t thresholdPicking;
    uint16_t thresholdOneBee;
    uint16_t thresholdTwoBees;
} Threshold2_t;

typedef struct{
    Point_t coordinates;
    float angle;
} Bee_t;

typedef enum{
    ONE_POINT,
    ALIVE,
    WAITING,
    LOST
} TrajectoryStatus_t;

typedef struct{
    TrajectoryStatus_t status;
    uint32_t waitStatesCount;
    std::vector<Bee_t> points;
} Trajectory_t;

typedef struct{
    uint32_t in;
    uint32_t out;
    uint32_t lost;
    uint32_t lostIn;
    uint32_t lostOut;
    uint32_t flyByIn;
    uint32_t flyByOut;
} OutputStatistics_t;

typedef struct{
    bool logBees;
    bool logTrajectories;
    bool generateOutputFrames;
    bool generateBeeImages;
    bool generateIntermediateImages;
} DebugFlags_t;

typedef struct{
    uint32_t inputIndex;
    std::string name;
    std::string framesPath;
    uint32_t baseRow;
    uint32_t baseCol;
    uint32_t resolutionDivider;
    FrameResolution_t inputResolution;
    FrameResolution_t usedResolution;
    std::vector<Rectangle_t> entranceRectangles;
    std::vector<Rectangle_t> borderRectangles;
} Input_t;

// performance counters

typedef struct {
    uint32_t frameIndex;
    float T1Timing;
    float T2Timing;
    float T3Timing;
    float T4Timing;
    float T5Timing;
    float T6Timing;
    float extractionTiming;
    float filteringTiming;
    float waitingTiming;
    float step1Timing;
} Step1InternalTimings_t;

typedef struct {
    uint32_t frameIndex;
    float subStep1Timing;
    float subStep2Timing;
    float subStep3Timing;
    float waitingTiming;
    float step2Timing;
} Step2InternalTimings_t;

typedef struct {
    uint32_t frameIndex;
    float step1Timing;
    float step2Timing;
    float step3Timing;
    float frameTiming;
} FrameTimings_t;

typedef struct {
    float allCoresTiming;
    float realTiming;
} TotalTiming_t;

#endif //EXERCISE5_STRUCTURES_HPP
