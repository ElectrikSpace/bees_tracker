//
// Created by sylvain on 30/01/23.
//

#ifndef EXERCISE5_STEP1_SIMPLIFIED_HPP
#define EXERCISE5_STEP1_SIMPLIFIED_HPP

#include "FrameLoader.hpp"
#include "Parameters.hpp"
#include "Logger.hpp"
#include "GlobalConcurrency.hpp"

#include <vector>
#include <filesystem>

/**
 * Simplified and vectorized implementation of Step 1
 */
class Step1Simplified {
private:
    Logger& logger;
    FrameLoader& frameLoader;
    GlobalConcurrency &concurrencyManager;

    // intermediate buffers
    std::vector<int16_t> T2;
    std::vector<int16_t> T2Bis;
    std::vector<uint16_t> T6;

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
     * constructor
     * @param logger reference to the logger
     * @param frameLoader reference to the frames loader
     * @param concurrencyManager reference to the concurrency manager
     */
    explicit Step1Simplified(Logger& logger, FrameLoader& frameLoader, GlobalConcurrency &concurrencyManager);

    /**
     * get class name
     * @return class name
     */
    static std::string getClassName() {return "Step1Simplified";}

    /**
     * process a frame
     * @param frameIndex index of frame to process
     * @return unique pointer to a vector representing the detected bees
     */
    std::unique_ptr<std::vector<Bee_t>> processFrame(uint32_t frameIndex);
};


#endif //EXERCISE5_STEP1_SIMPLIFIED_HPP
