//
// Created by sylvain on 30/01/23.
//

#ifndef EXERCISE5_LOGGER_HPP
#define EXERCISE5_LOGGER_HPP

#include <cstdint>
#include <vector>
#include <memory>
#include <atomic>

#include "Parameters.hpp"
#include "FrameLoader.hpp"

/**
 * Class responsible for logging lots of things and generate result files
 */
class Logger {
private:
    std::shared_ptr<Input_t> input;
    std::string schedulingPolicy;
    std::string step1Class;
    std::string step2Class;

    // logs related to the algorithm
    std::vector<std::unique_ptr<std::vector<Bee_t>>> beesLogs;
    std::vector<std::unique_ptr<std::vector<Trajectory_t>>> trajectoriesLogs;
    std::vector<std::unique_ptr<std::vector<Trajectory_t>>> lostTrajectoriesLogs;
    std::vector<OutputStatistics_t> statsLogs;

    // logs related to performance metrics
    std::atomic<uint32_t> beesLogsFrameIndex;
    std::atomic<uint32_t> trajectoriesLogsFrameIndex;
    std::atomic<uint32_t> lostTrajectoriesLogsFrameIndex;
    std::atomic<uint32_t> statsLogsFrameIndex;

    // performance

    uint32_t runsCounter;
    std::vector<std::pair<uint32_t, Step1InternalTimings_t>> step1InternalTimingLogs;
    std::vector<std::pair<uint32_t, Step2InternalTimings_t>> step2InternalTimingLogs;
    std::vector<std::pair<uint32_t, FrameTimings_t>> frameTimingLogs;
    std::vector<std::pair<uint32_t, TotalTiming_t>> totalTimingLogs;
    std::mutex step1Timing, step2Timing, frameTiming;

    /**
     * write the metadata file
     * @param path path to the output directory
     */
    void writeMetadataFile(std::string &path);

public:
    /**
     * constructor
     * @param input shared pointer to the input
     * @param step1ClassName step 1 implementation
     * @param step2ClassName step 2 implementation
     */
    explicit Logger(std::shared_ptr<Input_t> input, std::string step1ClassName, std::string step2ClassName);

    /**
     * reset the logger
     * @param newInput shared pointer to rhe new input
     * @param step1ClassName new step 1 implementation
     * @param step2ClassName new step 2 implementation
     * @param runIndex new run counter value
     */
    void reset(std::shared_ptr<Input_t> newInput, std::string step1ClassName, std::string step2ClassName, uint32_t runIndex);

    /**
     * set the scheduling policy
     * @param policy scheduling policy
     */
    void setSchedulingPolicy(std::string policy);

    /**
     * push the detected bees
     * @param bees unique pointer to the detected bees
     * @param frameIndex index of the related frame
     */
    void pushBees(std::unique_ptr<std::vector<Bee_t>> bees, uint32_t frameIndex);

    /**
     * push the current trajectories at the specified frame
     * @param currentTrajectories current trajectories at the specified frame
     * @param frameIndex index of the related frame
     */
    void pushCurrentTrajectories(std::vector<Trajectory_t> &currentTrajectories, uint32_t frameIndex);

    /**
     * push the lost trajectories at the specified frame
     * @param lostTrajectories lost trajectories at the specified frame
     * @param frameIndex index of the related frame
     */
    void pushLostTrajectories(std::unique_ptr<std::vector<Trajectory_t>> lostTrajectories, uint32_t frameIndex);

    /**
     * push output statistics at the specified frame
     * @param stats output statistics at the specified frame
     * @param frameIndex index of the related frame
     */
    void pushOutputStats(OutputStatistics_t& stats, uint32_t frameIndex);

    /**
     * write output files
     * @param path path to the output directory
     * @param frameLoader reference to the frames loader
     */
    void writeOutputFiles(std::string path, FrameLoader& frameLoader);

    // performance logs

    /**
     * increment the run counter
     */
    void incrementRunsCounter();

    /**
     * push internal timings of step 1
     * @param step1InternalTimings reference to internal timingq of step 1
     */
    void pushStep1InternalTimings(Step1InternalTimings_t& step1InternalTimings);

    /**
     * push internal timings of step 2
     * @param step2InternalTimings reference to internal timings of step 2
     */
    void pushStep2InternalTimings(Step2InternalTimings_t& step2InternalTimings);

    /**
     * push frame timings
     * @param frameTimings reference to frame timings
     */
    void pushFrameTimings(FrameTimings_t& frameTimings);

    /**
     * push total timings
     * @param totalTiming reference to total timings
     */
    void pushTotalTimings(TotalTiming_t& totalTiming);

    /**
     * write benchmark files
     * @param path path of the benchmark results directory
     * @param filePrefix prefix to add on each written file
     * @param append true if append to files is allowed
     * @param step1Internal true it internal timings of step 1 has to be written
     * @param step2Internal true it internal timings of step 2 has to be written
     * @param frames true if frames timings has to be written
     * @param total true if total timings has to be written
     * @param writeHeader true if header can be written, false if it's not allowed
     */
    void writeBenchmarkFiles(std::string& path, std::string filePrefix, bool append, bool step1Internal,
                             bool step2Internal, bool frames, bool total, bool writeHeader);
};


#endif //EXERCISE5_LOGGER_HPP
