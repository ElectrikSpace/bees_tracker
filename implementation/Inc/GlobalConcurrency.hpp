//
// Created by sylvain on 15/02/23.
//

#ifndef EXERCISE5_GLOBAL_CONCURRENCY_HPP
#define EXERCISE5_GLOBAL_CONCURRENCY_HPP

#include <vector>
#include <atomic>

#include "Parameters.hpp"
#include "FrameLoader.hpp"

/**
 * Class responsible for the management of concurrency and shared structures between frames
 */
class GlobalConcurrency {
private:
    // variable used to manage concurrency
    std::size_t frameAccumulatorBlocSize;
    uint32_t frameCount;
    std::vector<std::unique_ptr<std::atomic<uint32_t>>> framesAccumulatorIndex;
    std::atomic<uint32_t> step2Index;
    std::atomic<uint32_t> loggerIndex;
    std::atomic<uint32_t> currentFrameIndex;
    std::atomic<uint8_t> accumulatorReady;

    // shared structures
    std::vector<int16_t> framesAccumulator;
    std::vector<Trajectory_t> trackedBees;

public:
    /**
     * constructor
     * @param frameResolution input frame resolution
     * @param frameCount input frame count
     */
    GlobalConcurrency(FrameResolution_t frameResolution, uint32_t frameCount);

    // concurrency management

    /**
     * get the number of blocs dividing the frame accumulator
     * @return number of blocs dividing the frame accumulator
     */
    static std::size_t getFrameAccumulatorBlocCount();

    /**
     * get the size of blocs dividing the frame accumulator
     * @return size of blocs dividing the frame accumulator
     */
    std::size_t getFrameAccumulatorBlocSize();

    /**
     * prepare the accumulator before running the algorithm
     * @param frameLoader reference to the frame loader
     * @param step1ClassName name of Step 1 implementation
     */
    void prepareAccumulator(FrameLoader& frameLoader, std::string step1ClassName);

    /**
     * lock an accumulator bloc
     * @param index index of the bloc
     * @param frameIndex index of the working frame
     */
    void lockAccumulatorBloc(uint32_t index, uint32_t frameIndex);

    /**
     * unlock an accumulator bloc
     * @param index index of the bloc
     */
    void unlockAccumulatorBloc(uint32_t index);

    /**
     * lock step 2
     * @param frameIndex index of the working frame
     */
    void lockStep2(uint32_t frameIndex);

    /**
     * unlock step 2
     */
    void unlockStep2();

    /**
     * request a new frame to process
     * @param frameIndex reference to the frame index, written if a new frame is available
     * @return true if a new frame is available, false otherwise
     */
    bool requestNewFrameIndex(uint32_t &frameIndex);

    // shared structures access

    /**
     * get the frame accumulator
     * @return a reference to the frame accumulator
     */
    std::vector<int16_t>& getFramesAccumulator();

    /**
     * get the tracked bees
     * @return a references to the tracked bees
     */
    std::vector<Trajectory_t>& getTrackedBees();
};


#endif //EXERCISE5_GLOBAL_CONCURRENCY_HPP
