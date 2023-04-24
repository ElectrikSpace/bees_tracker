//
// Created by sylvain on 30/01/23.
//

#ifndef EXERCISE5_STEP2_PARALLEL_HPP
#define EXERCISE5_STEP2_PARALLEL_HPP

#include <vector>

#include "Parameters.hpp"
#include "WorkManager.hpp"

/**
 * Parallel implementation of Step 2
 */
class Step2Parallel {
private:
    Logger& logger;
    GlobalConcurrency &concurrencyManager;

public:
    /**
     * constructor
     * @param logger reference to the logger
     * @param concurrencyManager reference to the concurrency manager
     */
    Step2Parallel(Logger& logger, GlobalConcurrency &concurrencyManager);

    /**
     * get class name
     * @return class name
     */
    static std::string getClassName() {return "Step2Sequential";}

    /**
     * update the bee trajectories
     * @param bees unique pointer to the detected bees
     * @param frameIndex index of the related frame
     * @return unique pointer to the lost trajectory at this frame
     */
    std::unique_ptr<std::vector<Trajectory_t>> updateBeeTrajectories(std::unique_ptr<std::vector<Bee_t>> bees,
                                                                     uint32_t frameIndex);
};


#endif //EXERCISE5_STEP2_PARALLEL_HPP
