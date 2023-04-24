//
// Created by sylvain on 29/01/23.
//

#ifndef EXERCISE5_STEP1_NAIVE_HPP
#define EXERCISE5_STEP1_NAIVE_HPP

#include "FrameLoader.hpp"
#include "Parameters.hpp"
#include "Logger.hpp"
#include "GlobalConcurrency.hpp"

#include <vector>
#include <filesystem>

/**
 * Naive implementation of Step 1
 */
class Step1Naive {
private:
    Logger& logger;
    FrameLoader& frameLoader;
    GlobalConcurrency& concurrencyManager;

    // intermediate buffers
    std::vector<int16_t> T1;
    std::vector<int16_t> T2;
    std::vector<int16_t> T3;
    std::vector<uint16_t> T4;
    std::vector<uint16_t> T5;
    std::vector<uint16_t> T6;

    /**
     * execute sub-step 1: do subtraction and update frames accumulator
     * @param frameIndex index of frame to process
     * @return waiting time (performance counter used for benchmarking)
     */
    float subStep1(uint32_t frameIndex);

    /**
     * execute sub-step 2: assemble RGB channels
     */
    void subStep2();

    /**
     * execute sub-step 3: apply 5x5 blur
     */
    void subStep3();

    /**
     * execute sub-step 4: apply threshold 1
     */
    void subStep4();

    /**
     * execute sub-step 5: reduce
     */
    void subStep5();

    /**
     * execute sub-step 6: apply 3x3 blur
     */
    void subStep6();

    /**
     * execute sub-step 7: extract raw bees
     * @return unique pointer to raw bees
     */
    std::unique_ptr<std::vector<std::unique_ptr<std::vector<Point_t>>>> subStep7();

    /**
     * execute sub-step 8: compute coordinate and angle of each bee
     * @param beesPoints raw bees
     * @return unique pointer to bees
     */
    std::unique_ptr<std::vector<Bee_t>> subStep8(std::unique_ptr<std::vector<std::unique_ptr<std::vector<Point_t>>>> beesPoints);

    /**
     * generate small bee images
     * @param bees reference to the vector of detected bees
     * @param frameIndex index of related frame
     */
    void generateBeeImages(std::vector<Bee_t>& bees, uint32_t frameIndex);

    /**
     * generate images of intermediate buffers
     * @param frameIndex index of related frame
     */
    void generateIntermediateImages(uint32_t frameIndex);

public:
    /**
     * constructor
     * @param logger reference to the logger
     * @param frameLoader reference to the frames loader
     * @param concurrencyManager reference to the concurrency manager
     */
    explicit Step1Naive(Logger& logger, FrameLoader& frameLoader, GlobalConcurrency& concurrencyManager);

    /**
     * get class name
     * @return class name
     */
    static std::string getClassName() {return "Step1Naive";}

    /**
     * process a frame
     * @param frameIndex index of frame to process
     * @return unique pointer to a vector representing the detected bees
     */
    std::unique_ptr<std::vector<Bee_t>> processFrame(uint32_t frameIndex);
};


#endif //EXERCISE5_STEP1_NAIVE_HPP
