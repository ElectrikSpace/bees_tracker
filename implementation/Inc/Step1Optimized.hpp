//
// Created by sylvain on 30/01/23.
//

#ifndef EXERCISE5_STEP1_OPTIMIZED_HPP
#define EXERCISE5_STEP1_OPTIMIZED_HPP

#include "FrameLoader.hpp"
#include "Parameters.hpp"
#include "Logger.hpp"
#include "GlobalConcurrency.hpp"

#include <vector>
#include <filesystem>

/**
 * Optimized but still scalar implementation of step 1
 */
class Step1Optimized {
private:
    Logger& logger;
    FrameLoader& frameLoader;
    GlobalConcurrency& concurrencyManager;

    // intermediate buffers
    std::vector<int16_t> T2;
    std::vector<int16_t> T3;
    std::vector<uint16_t> T5;
    std::vector<uint16_t> T6;

    /**
     * execute sub-step 1 and 2: do subtraction, update frames accumulator, and assemble RGB channels
     * @param frameIndex index of frame to process
     * @return waiting time (performance counter used for benchmarking)
     */
    float subStep1And2(uint32_t frameIndex);

    /**
     * execute sub-step 3: apply 5x5 blur
     */
    void subStep3();

    /**
     * execute sub-step 4 and 5: apply 5x5 blur and reduce
     */
    void subStep4And5();

    /**
     * execute sub-step 6: apply 3x3 blur
     */
    void subStep6();

    /**
     * execute sub-step 7: extract raw bees
     * @return unique pointer to raw bees
     */
    std::unique_ptr<std::vector<Bee_t>> subStep7And8();

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
     *
     * @param logger
     * @param frameLoader
     * @param concurrencyManager
     */
    explicit Step1Optimized(Logger& logger, FrameLoader& frameLoader, GlobalConcurrency& concurrencyManager);

    /**
     * get class name
     * @return class name
     */
    static std::string getClassName() {return "Step1Optimized";}

    /**
     * process a frame
     * @param frameIndex index of frame to process
     * @return unique pointer to a vector representing the detected bees
     */
    std::unique_ptr<std::vector<Bee_t>> processFrame(uint32_t frameIndex);
};


#endif //EXERCISE5_STEP1_OPTIMIZED_HPP
