//
// Created by sylvain on 29/01/23.
//

#ifndef EXERCISE5_STEP3_HPP
#define EXERCISE5_STEP3_HPP

#include <memory>
#include <utility>

#include "Parameters.hpp"
#include "Logger.hpp"
#include "GlobalConcurrency.hpp"

/**
 * Implementation of Step 3
 */
class Step3 {
private:
    Logger& logger;
    Input_t& input;

public:
    /**
     * constructor
     * @param logger reference to the logger
     * @param input reference to the input
     */
    explicit Step3(Logger& logger, Input_t& input);

    /**
     * process lost trajectories and update statistics
     * @param trajectories unique pointer to the lost trajectories
     * @param frameIndex index of related frame
     */
    void processLostTrajectories(std::unique_ptr<std::vector<Trajectory_t>> trajectories, uint32_t frameIndex);
};


#endif //EXERCISE5_STEP3_HPP
