//
// Created by sylvain on 25/01/23.
//

#ifndef EXERCISE5_PARAMETERS_HPP
#define EXERCISE5_PARAMETERS_HPP

#include <utility>
#include <cstdint>
#include <vector>
#include <memory>

#include "Structures.hpp"

/**
 * global parameters
 */
class Parameters {
public:
    // parameters for step 1
    static inline std::size_t STORED_FRAMES = 32;
    static inline std::size_t STORED_FRAMES_LOG2 = 5;
    static constexpr Pixel_t COLORS_WEIGHTS = {1, 3, 4};
    static constexpr uint32_t COLORS_WEIGHTS_LOG2 = 3;
    static constexpr Threshold1_t THRESHOLD_1 = {-400, 400, -2};
    static constexpr std::size_t N_REDUCE = 2;
    static constexpr Threshold2_t THRESHOLD_2 = {220, 20, 110};
    static inline std::size_t STEP1_ACCUMULATOR_BLOC_COUNT = 16;
    static inline bool USE_GREY_SCALE_FRAMES_ACCUMULATOR = false;

    // parameters for step 2
    static inline float PATH_DETECTION_RADIUS = 150.0;
    static inline float SINGLE_BEE_DETECTION_RADIUS = 300.0;
    static inline std::size_t PATH_MAX_WAIT = 2;

    // parameters for step 3 are included in the input structure

    // other parameters
//    static constexpr DebugFlags_t DEBUG_FLAGS = {false, false, false, false, false};
//    static constexpr DebugFlags_t DEBUG_FLAGS = {false, false, true, true, false};
    static constexpr DebugFlags_t DEBUG_FLAGS = {true, true, true, true, true};
    static constexpr bool ENABLE_PERFORMANCE_LOG = true;
    static constexpr uint32_t BEE_IMAGE_SIZE = 64;
    static inline std::size_t CORE_COUNT = 4;
    static inline std::size_t MAX_THREADS_PER_FRAME = 0; // set by work scheduler
    static inline std::size_t MAX_FRAMES_IN_PARALLEL = 0; // set by work scheduler
};

#endif //EXERCISE5_PARAMETERS_HPP
