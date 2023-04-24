//
// Created by sylvain on 31/01/23.
//

#ifndef EXERCISE5_WORK_MANAGER_HPP
#define EXERCISE5_WORK_MANAGER_HPP

#include <string>
#include <memory>
#include <thread>
//#include <barrier>
#include <utility>

#include "Logger.hpp"
#include "FrameLoader.hpp"
#include "GlobalConcurrency.hpp"

/**
 * Class responsible for global work scheduling
 * @tparam S1 step 1 implementation
 * @tparam S2 step 2 implementation
 * @tparam S3 step 3 implementation
 */
template<class S1, class S2, class S3>
class WorkManager {
private:
    std::shared_ptr<FrameLoader> frameLoader;
    std::shared_ptr<Input_t> input;
    GlobalConcurrency concurrencyManager;

    /**
     * run the algorithm
     * @param logger reference to the logger
     * @param policy policy name
     * @return elapsed time
     */
    float run(const std::shared_ptr<Logger>& logger, std::string policy);

public:
    /**
     * constructor with internal frames loader
     * @param input reference to the input
     * @param frameCount number of frames to process
     */
    WorkManager(const std::shared_ptr<Input_t>& input, std::size_t frameCount);

    /**
     * constructor with external frames loader
     * @param input reference to the input
     * @param externalFrameLoader shared pointer to an externally created frames loader
     */
    WorkManager(const std::shared_ptr<Input_t>& input, std::shared_ptr<FrameLoader> externalFrameLoader);

    /**
     * get frames loader
     * @return shared pointer to the frames loader
     */
    std::shared_ptr<FrameLoader> getFrameLoader();

    /**
     * run the algorithm using all available cores to process each frame sequentially
     * @param logger reference to the logger
     * @return elapsed time
     */
    float runSimpleMulticore(const std::shared_ptr<Logger>& logger);

    /**
     * run the algorithm using one core for each frame
     * @param logger reference to the logger
     * @return elapsed time
     */
    float runOneFramePerCore(const std::shared_ptr<Logger>& logger);

    /**
     * run the algorithm using an hybrid approach between runSimpleMulticore() and runOneFramePerCore()
     * @param logger reference to the logger
     * @return elapsed time
     */
    float runHybrid(const std::shared_ptr<Logger>& logger);
};

template<class S1, class S2, class S3>
WorkManager<S1, S2, S3>::WorkManager(const std::shared_ptr<Input_t> &input, std::size_t frameCount)
        : concurrencyManager(input->usedResolution, frameCount){
    this->input = input;
    frameLoader = std::make_shared<FrameLoader>(*input, frameCount);
}

template<class S1, class S2, class S3>
WorkManager<S1, S2, S3>::WorkManager(const std::shared_ptr<Input_t> &input, std::shared_ptr<FrameLoader> externalFrameLoader)
        : concurrencyManager(input->usedResolution, externalFrameLoader->getFrameCount()){
    this->input = input;
    frameLoader = std::move(externalFrameLoader);
}

template<class S1, class S2, class S3>
float WorkManager<S1, S2, S3>::run(const std::shared_ptr<Logger> &logger, std::string policy) {
    logger->setSchedulingPolicy(policy);

    concurrencyManager.prepareAccumulator(*frameLoader, S1::getClassName());
    concurrencyManager.getTrackedBees().clear();

    std::atomic<float> allCoresTiming = 0;
    auto g0 = std::chrono::steady_clock::now();

//    std::barrier initializationDone(Parameters::MAX_FRAMES_IN_PARALLEL, [&] {
//        g0 = std::chrono::steady_clock::now();
//    });

    const auto work = [&] {
        uint32_t frame;

        // instantiate the three steps
        auto step1 = std::make_unique<S1>(*logger, *frameLoader, concurrencyManager);
        auto step2 = std::make_unique<S2>(*logger, concurrencyManager);
        auto step3 = std::make_unique<S3>(*logger, *input);

//        initializationDone.arrive_and_wait();

        auto c0 = std::chrono::steady_clock::now();

        while (concurrencyManager.requestNewFrameIndex(frame)) {
            auto t0 = std::chrono::steady_clock::now();

            std::unique_ptr<std::vector<Bee_t>> bees = step1->processFrame(frame);

            auto t1 = std::chrono::steady_clock::now();

            std::unique_ptr<std::vector<Trajectory_t>> lostTrajectories = step2->updateBeeTrajectories(std::move(bees), frame);

            auto t2 = std::chrono::steady_clock::now();

            step3->processLostTrajectories(std::move(lostTrajectories), frame);

            if (Parameters::ENABLE_PERFORMANCE_LOG) {
                auto t3 = std::chrono::steady_clock::now();
                FrameTimings_t t;
                t.frameIndex = frame;
                t.step1Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / 1000;
                t.step2Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000;
                t.step3Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count()) / 1000;
                t.frameTiming = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t3 - t0).count()) / 1000;
                logger->pushFrameTimings(t);
            }
        }

        if (Parameters::ENABLE_PERFORMANCE_LOG) {
            auto c1 = std::chrono::steady_clock::now();
            float timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(c1 - c0).count()) / 1000;
            allCoresTiming.fetch_add(timing);
        }
    };

    // create threads
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < Parameters::MAX_FRAMES_IN_PARALLEL; i++)
        threads.emplace_back(work);

    // wait until all threads have finished
    for (std::size_t i = 0; i < Parameters::MAX_FRAMES_IN_PARALLEL; i++)
        threads[i].join();

    auto g1 = std::chrono::steady_clock::now();
    float realTiming = ((float) std::chrono::duration_cast<std::chrono::microseconds>(g1 - g0).count()) / 1000;
//    std::cout << policy << ": " << std::endl;
//    std::cout << "  -> real time = " << realTiming << " ms" << std::endl;
//    std::cout << "  -> work time = " << allCoresTiming << " ms" << std::endl;

    if (Parameters::ENABLE_PERFORMANCE_LOG) {
        TotalTiming_t t;
        t.allCoresTiming = allCoresTiming;
        t.realTiming = realTiming;
        logger->pushTotalTimings(t);
    }

    return realTiming;
}

template<class S1, class S2, class S3>
std::shared_ptr<FrameLoader> WorkManager<S1, S2, S3>::getFrameLoader() {
    return frameLoader;
}

template<class S1, class S2, class S3>
float WorkManager<S1, S2, S3>::runSimpleMulticore(const std::shared_ptr<Logger>& logger) {
    Parameters::MAX_FRAMES_IN_PARALLEL = 1;
    Parameters::MAX_THREADS_PER_FRAME = Parameters::CORE_COUNT;

    return run(logger, "simpleMultiCore");
}

template<class S1, class S2, class S3>
float WorkManager<S1, S2, S3>::runOneFramePerCore(const std::shared_ptr<Logger>& logger) {
    Parameters::MAX_FRAMES_IN_PARALLEL = Parameters::CORE_COUNT;
    Parameters::MAX_THREADS_PER_FRAME = 1;

    return run(logger, "oneFramePerCore");
}

template<class S1, class S2, class S3>
float WorkManager<S1, S2, S3>::runHybrid(const std::shared_ptr<Logger>& logger) {
    if ((Parameters::CORE_COUNT % 2) && (Parameters::CORE_COUNT != 1))
        throw std::logic_error("error: CORE_COUNT must be a power of 2");

    Parameters::MAX_FRAMES_IN_PARALLEL = 1;
    Parameters::MAX_THREADS_PER_FRAME = 1;
    if (Parameters::CORE_COUNT > 1) {
        Parameters::MAX_FRAMES_IN_PARALLEL = Parameters::CORE_COUNT / 2;
        Parameters::MAX_THREADS_PER_FRAME = 2;
    }

    return run(logger, "hybrid");
}

#endif //EXERCISE5_WORK_MANAGER_HPP
