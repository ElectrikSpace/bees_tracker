//
// Created by sylvain on 29/01/23.
//

#ifndef EXERCISE5_STEP2_SEQUENTIAL_HPP
#define EXERCISE5_STEP2_SEQUENTIAL_HPP

#include <utility>
#include <vector>

#include "Parameters.hpp"
#include "Logger.hpp"
#include "GlobalConcurrency.hpp"

/**
 * Sequential implementation of Step 2
 */
class Step2Sequential {
private:
    Logger& logger;
    GlobalConcurrency &concurrencyManager;

public:
    /**
     * constructor
     * @param logger reference to the logger
     * @param concurrencyManager reference to the concurrency manager
     */
    Step2Sequential(Logger& logger, GlobalConcurrency &concurrencyManager);

    /**
     * get class name
     * @return class name
     */
    static std::string getClassName() {return "Step2Parallel";}

    /**
     * update the bee trajectories
     * @param bees unique pointer to the detected bees
     * @param frameIndex index of the related frame
     * @return unique pointer to the lost trajectory at this frame
     */
    std::unique_ptr<std::vector<Trajectory_t>> updateBeeTrajectories(std::unique_ptr<std::vector<Bee_t>> bees,
                                                                     uint32_t frameIndex);
};


#endif //EXERCISE5_STEP2_SEQUENTIAL_HPP
