//
// Created by sylvain on 31/01/23.
//

#ifndef EXERCISE5_INPUT_COLLECTION_HPP
#define EXERCISE5_INPUT_COLLECTION_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

#include "Parameters.hpp"

/**
 * Input collection: all methods are static
 */
class InputCollection {
public:
    /**
     * get the number of available inputs
     * @return number of available input
     */
    static std::size_t getInputsCount();

    /**
     * get the number of frame within an input
     * @param inputIndex index of the input
     * @return number of frames
     */
    static std::size_t getInputFrameCount(uint32_t inputIndex);

    /**
     * create an input structure using some parameters
     * @param inputIndex index of this input
     * @param resolutionDivisor divider to apply to the default resolution of the input
     * @return shared pointer to the generated input structure
     */
    static std::shared_ptr<Input_t> createInput(uint32_t inputIndex, uint32_t resolutionDivisor);
};

#endif //EXERCISE5_INPUT_COLLECTION_HPP
