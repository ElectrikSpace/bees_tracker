//
// Created by sylvain on 15/02/23.
//

#include "GlobalConcurrency.hpp"
#include "Step1Simplified.hpp"

GlobalConcurrency::GlobalConcurrency(FrameResolution_t frameResolution, uint32_t frameCount)
: framesAccumulator((Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR ? 1 : 3) * frameResolution.rows*frameResolution.cols, 0) {
    uint32_t elementCount = frameResolution.rows*frameResolution.cols;
    frameAccumulatorBlocSize = elementCount / (Parameters::STEP1_ACCUMULATOR_BLOC_COUNT - 1);
    for (uint32_t i = 0; i < Parameters::STEP1_ACCUMULATOR_BLOC_COUNT; i++) {
        framesAccumulatorIndex.push_back(std::make_unique<std::atomic<uint32_t>>());
        framesAccumulatorIndex[i]->store(0);
    }

    step2Index.store(0);
    loggerIndex.store(0);
    currentFrameIndex.store(0);
    accumulatorReady.store(0);
    this->frameCount = frameCount;
}

std::size_t GlobalConcurrency::getFrameAccumulatorBlocCount() {
    return Parameters::STEP1_ACCUMULATOR_BLOC_COUNT;
}

std::size_t GlobalConcurrency::getFrameAccumulatorBlocSize() {
    return frameAccumulatorBlocSize;
}

void GlobalConcurrency::lockAccumulatorBloc(uint32_t index, uint32_t frameIndex) {
    while (framesAccumulatorIndex[index]->load() < frameIndex);
}

void GlobalConcurrency::unlockAccumulatorBloc(uint32_t index) {
    framesAccumulatorIndex[index]->fetch_add(1);
}

void GlobalConcurrency::lockStep2(uint32_t frameIndex) {
    while (step2Index.load() < frameIndex);
}

void GlobalConcurrency::unlockStep2() {
    step2Index.fetch_add(1);
}

bool GlobalConcurrency::requestNewFrameIndex(uint32_t &frameIndex) {
    if (currentFrameIndex.load() >= frameCount)
        return false;

    frameIndex = currentFrameIndex.fetch_add(1);

    if (frameIndex >= frameCount)
        return false;

    return true;
}

std::vector<int16_t> &GlobalConcurrency::getFramesAccumulator() {
    return framesAccumulator;
}

std::vector<Trajectory_t> &GlobalConcurrency::getTrackedBees() {
    return trackedBees;
}

void GlobalConcurrency::prepareAccumulator(FrameLoader &frameLoader, std::string step1ClassName) {
    // initialize accumulator
    uint32_t elementsCount = frameLoader.getFramesResolution().rows*frameLoader.getFramesResolution().cols;
    elementsCount = (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) ? elementsCount : 3*elementsCount;
    for (uint32_t frame = 0; frame < Parameters::STORED_FRAMES; frame++) {
        uint8_t* currentFrame = frameLoader.getInitializationFramePtr(frame);
        if (step1ClassName == Step1Simplified::getClassName() && !Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
            for (uint32_t k = 0; k < Parameters::STEP1_ACCUMULATOR_BLOC_COUNT; k++) {
                uint32_t start = 3 * k * frameAccumulatorBlocSize;
                uint32_t stop = ((k + 1) * 3 * frameAccumulatorBlocSize <= elementsCount)
                                ? (k + 1) * 3 * frameAccumulatorBlocSize : elementsCount;
                uint32_t simdStop = stop - ((stop-start) % 24);
                for (uint32_t i = start; i < simdStop; i += 24) {
                    for (uint32_t j = 0; j < 8; j++) {
                        framesAccumulator[i + j] = (int16_t) (framesAccumulator[i + j] + currentFrame[i + 3 * j]); // R
                        framesAccumulator[i + j + 8] = (int16_t) (framesAccumulator[i + j + 8] +
                                                                  currentFrame[i + 3 * j + 1]); // G
                        framesAccumulator[i + j + 16] = (int16_t) (framesAccumulator[i + j + 16] +
                                                                   currentFrame[i + 3 * j + 2]); // B
                    }
                }
                for (uint32_t i = simdStop; i < stop; i++)
                    framesAccumulator[i] = (int16_t) (framesAccumulator[i] + currentFrame[i]);
            }
        } else {
            for (uint32_t i = 0; i < elementsCount; i++) {
                if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                    int32_t value = 0;
                    value += currentFrame[3*i]*Parameters::COLORS_WEIGHTS.r;
                    value += currentFrame[3*i + 1]*Parameters::COLORS_WEIGHTS.g;
                    value += currentFrame[3*i + 2]*Parameters::COLORS_WEIGHTS.b;
                    value >>= Parameters::COLORS_WEIGHTS_LOG2;
                    framesAccumulator[i] = (int16_t) (framesAccumulator[i] + value);
                }
                else
                    framesAccumulator[i] = (int16_t) (framesAccumulator[i] + currentFrame[i]);
            }
        }
    }
}

