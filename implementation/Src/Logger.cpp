//
// Created by sylvain on 30/01/23.
//

#include "Logger.hpp"
#include "CSVWriter.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <filesystem>
#include <utility>

#if __has_include(<jsoncpp/json/value.h>)
    // path for debian
    #include <jsoncpp/json/value.h>
    #include <jsoncpp/json/writer.h>
#else
    // path for arch linux
    #include <json/value.h>
    #include <json/writer.h>
#endif

Logger::Logger(std::shared_ptr<Input_t> input, std::string step1ClassName, std::string step2ClassName) {
    this->input = std::move(input);
    step1Class = std::move(step1ClassName);
    step2Class = std::move(step2ClassName);
    runsCounter = 0;
}

void Logger::setSchedulingPolicy(std::string policy) {
    schedulingPolicy = std::move(policy);
}

void Logger::pushBees(std::unique_ptr<std::vector<Bee_t>> bees, uint32_t frameIndex)
{
    while (beesLogsFrameIndex.load() < frameIndex);

    beesLogs.push_back(std::move(bees));

    beesLogsFrameIndex.fetch_add(1);
}

void Logger::pushCurrentTrajectories(std::vector<Trajectory_t> &currentTrajectories, uint32_t frameIndex)
{
    while (trajectoriesLogsFrameIndex.load() < frameIndex);

    auto trajectoriesCopy = std::make_unique<std::vector<Trajectory_t>>(currentTrajectories.size());
    Trajectory_t* ptr = trajectoriesCopy->data();
    for (uint32_t i = 0; i < currentTrajectories.size(); i++)
        ptr[i] = currentTrajectories[i];

    trajectoriesLogs.push_back(std::move(trajectoriesCopy));

    trajectoriesLogsFrameIndex.fetch_add(1);
}

void Logger::pushLostTrajectories(std::unique_ptr<std::vector<Trajectory_t>> lostTrajectories, uint32_t frameIndex)
{
    while (lostTrajectoriesLogsFrameIndex.load() < frameIndex);

    lostTrajectoriesLogs.push_back(std::move(lostTrajectories));

    lostTrajectoriesLogsFrameIndex.fetch_add(1);
}

void Logger::pushOutputStats(OutputStatistics_t& stats, uint32_t frameIndex)
{
    while (statsLogsFrameIndex.load() < frameIndex);

    statsLogs.push_back(stats);

    statsLogsFrameIndex.fetch_add(1);
}

void Logger::writeOutputFiles(std::string path, FrameLoader& frameLoader)
{
    // remove all previous files
    // create directory if it doesn't exist
    if (std::filesystem::is_directory(path))
        for (const auto& entry : std::filesystem::directory_iterator(path))
            std::filesystem::remove_all(entry.path());
    else
        std::filesystem::create_directory(path);

    // create metadata file
    writeMetadataFile(path);

    // create stats file
    CSVWriter csvStats;
    csvStats.newRow() << "frame" <<  "in" << "out" << "lost" << "lostIn" << "lostOut" << "flyByIn" << "flyByOut";
    uint32_t frame = 0;
    for (OutputStatistics_t& stat : statsLogs)
        csvStats.newRow() << frame++ << stat.in << stat.out << stat.lost << stat.lostIn << stat.lostOut
            << stat.flyByIn << stat.flyByOut;
    csvStats.writeToFile(path + "/output_statistics.csv", false);

    // create bees files if needed
    if (Parameters::DEBUG_FLAGS.logBees) {
        CSVWriter csvBees;
        csvBees.newRow() << "frame" << "BeeID" << "x" << "y" << "angle";
        for (frame = 0; frame < beesLogs.size(); frame++) {
            for (uint32_t i = 0; i < beesLogs[frame]->size(); i++) {
                Bee_t bee = (*beesLogs[frame])[i];
                csvBees.newRow() << frame << i << bee.coordinates.x << bee.coordinates.y << bee.angle;
            }
        }
        csvBees.writeToFile(path + "/bees.csv");
    }

    // create trajectories files if needed
    if (Parameters::DEBUG_FLAGS.logTrajectories) {
        CSVWriter csvTrajectories;
        csvTrajectories.newRow() << "frame" << "trajectory" << "status" << "waitCount" << "pointID" << "x" << "y" << "angle";
        for (frame = 0; frame < trajectoriesLogs.size(); frame++) {
            for (uint32_t t = 0; t < (*trajectoriesLogs[frame]).size(); t++) {
                Trajectory_t& trajectory = (*trajectoriesLogs[frame])[t];
                for (uint32_t i = 0; i < trajectory.points.size(); i++) {
                    csvTrajectories.newRow() << frame << t << trajectory.status << trajectory.waitStatesCount << i
                         << trajectory.points[i].coordinates.x << trajectory.points[i].coordinates.y << trajectory.points[i].angle;
                }
            }
        }
        csvTrajectories.writeToFile(path + "/trajectories.csv");

        CSVWriter csvLostTrajectories;
        csvLostTrajectories.newRow() << "frame" << "trajectory" << "status" << "waitCount" << "pointID" << "x" << "y";
        for (frame = 0; frame < lostTrajectoriesLogs.size(); frame++) {
            for (uint32_t t = 0; t < (*lostTrajectoriesLogs[frame]).size(); t++) {
                Trajectory_t& trajectory = (*lostTrajectoriesLogs[frame])[t];
                for (uint32_t i = 0; i < trajectory.points.size(); i++) {
                    csvLostTrajectories.newRow() << frame << t << trajectory.status << trajectory.waitStatesCount << i
                        << trajectory.points[i].coordinates.x << trajectory.points[i].coordinates.y;
                }
            }
        }
        csvLostTrajectories.writeToFile(path + "/lost_trajectories.csv");
    }

    // generate output frame if needed
    if (Parameters::DEBUG_FLAGS.generateOutputFrames) {
        std::filesystem::create_directory(path + "/output_frames");

        for (frame = 0; frame < beesLogs.size(); frame++) {
            cv::Mat img = frameLoader.getFrameMat(frame).clone();
            for (Bee_t &bee : *beesLogs[frame]) {
                cv::Point point(static_cast<int>(bee.coordinates.y),
                                static_cast<int>(bee.coordinates.x));
                cv::Scalar color(255, 0, 0);
                cv::circle(img, point, 50, color, 8);
            }
            cv::imwrite(path + "/output_frames/frame_" + std::to_string(frame) + ".jpg", img);
        }
    }

    // generate bee images if needed
    if (Parameters::DEBUG_FLAGS.generateBeeImages)
        std::filesystem::rename("tmp_bee_images", path + "/bee_images");

    // generate intermediate results if needed
    if (Parameters::DEBUG_FLAGS.generateIntermediateImages)
        std::filesystem::rename("intermediate_results", path + "/intermediate_results");
}

void Logger::writeMetadataFile(std::string &path) {
    Json::Value metadata;

    // add input
    metadata["input"]["inputIndex"] = input->inputIndex;
    metadata["input"]["name"] = input->name;
    metadata["input"]["framesPath"] = input->framesPath;
    metadata["input"]["baseRow"] = input->baseRow;
    metadata["input"]["baseCol"] = input->baseCol;
    metadata["input"]["resolutionDivider"] = input->resolutionDivider;
    metadata["input"]["usedResolution"]["rows"] = input->usedResolution.rows;
    metadata["input"]["usedResolution"]["cols"] = input->usedResolution.cols;
    Json::Value entranceRects(Json::arrayValue);
    for (Rectangle_t& r : input->entranceRectangles) {
        Json::Value rect;
        rect["start"]["x"] = r.p1.x;
        rect["start"]["y"] = r.p1.y;
        rect["stop"]["x"] = r.p2.x;
        rect["stop"]["y"] = r.p2.y;
        entranceRects.append(rect);
    }
    metadata["input"]["entranceRectangles"] = entranceRects;
    Json::Value borderRects(Json::arrayValue);
    for (Rectangle_t& r : input->borderRectangles) {
        Json::Value rect;
        rect["start"]["x"] = r.p1.x;
        rect["start"]["y"] = r.p1.y;
        rect["stop"]["x"] = r.p2.x;
        rect["stop"]["y"] = r.p2.y;
        borderRects.append(rect);
    }
    metadata["input"]["borderRectangles"] = borderRects;

    // add parameters
    metadata["parameters"]["STORED_FRAMES"] = Parameters::STORED_FRAMES;
    metadata["parameters"]["STORED_FRAMES_LOG2"] = Parameters::STORED_FRAMES_LOG2;
    metadata["parameters"]["COLORS_WEIGHTS"]["r"] = Parameters::COLORS_WEIGHTS.r;
    metadata["parameters"]["COLORS_WEIGHTS"]["g"] = Parameters::COLORS_WEIGHTS.g;
    metadata["parameters"]["COLORS_WEIGHTS"]["b"] = Parameters::COLORS_WEIGHTS.b;
    metadata["parameters"]["THRESHOLD_1"]["min"] = Parameters::THRESHOLD_1.thMin;
    metadata["parameters"]["THRESHOLD_1"]["max"] = Parameters::THRESHOLD_1.thMax;
    metadata["parameters"]["THRESHOLD_1"]["factor"] = Parameters::THRESHOLD_1.thFactor;
    metadata["parameters"]["N_REDUCE"] = Parameters::N_REDUCE;
    metadata["parameters"]["THRESHOLD_2"]["picking"] = Parameters::THRESHOLD_2.thresholdPicking;
    metadata["parameters"]["THRESHOLD_2"]["one_bee"] = Parameters::THRESHOLD_2.thresholdOneBee;
    metadata["parameters"]["THRESHOLD_2"]["two_bees"] = Parameters::THRESHOLD_2.thresholdTwoBees;
    metadata["parameters"]["STEP1_ACCUMULATOR_BLOC_COUNT"] = Parameters::STEP1_ACCUMULATOR_BLOC_COUNT;
    metadata["parameters"]["PATH_DETECTION_RADIUS"] = Parameters::PATH_DETECTION_RADIUS;
    metadata["parameters"]["SINGLE_BEE_DETECTION_RADIUS"] = Parameters::SINGLE_BEE_DETECTION_RADIUS;
    metadata["parameters"]["PATH_MAX_WAIT"] = Parameters::PATH_MAX_WAIT;
    metadata["parameters"]["BEE_IMAGE_SIZE"] = Parameters::BEE_IMAGE_SIZE;
    metadata["parameters"]["DEBUG_LEVEL"]["log_bees"] = Parameters::DEBUG_FLAGS.logBees;
    metadata["parameters"]["DEBUG_LEVEL"]["log_trajectories"] = Parameters::DEBUG_FLAGS.logTrajectories;
    metadata["parameters"]["DEBUG_LEVEL"]["generate_output_frames"] = Parameters::DEBUG_FLAGS.generateOutputFrames;
    metadata["parameters"]["DEBUG_LEVEL"]["generate_bee_images"] = Parameters::DEBUG_FLAGS.generateBeeImages;
    metadata["parameters"]["DEBUG_LEVEL"]["generate_intermediate_images"] = Parameters::DEBUG_FLAGS.generateIntermediateImages;
    metadata["parameters"]["CORE_COUNT"] = Parameters::CORE_COUNT;
    metadata["parameters"]["MAX_THREADS_PER_FRAME"] = Parameters::MAX_THREADS_PER_FRAME;
    metadata["parameters"]["MAX_FRAMES_IN_PARALLEL"] = Parameters::MAX_FRAMES_IN_PARALLEL;

    // add others
    metadata["scheduling"] = schedulingPolicy;

    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "   ";
    std::unique_ptr<Json::StreamWriter> writer(
            builder.newStreamWriter());
    std::ofstream metadataFile;
    metadataFile.open (path + "/metadata.json");
    writer->write(metadata, &metadataFile);
    metadataFile.close();
}

void Logger::reset(std::shared_ptr<Input_t> newInput, std::string step1ClassName, std::string step2ClassName, uint32_t runIndex) {
    this->input = std::move(newInput);
    step1Class = std::move(step1ClassName);
    step2Class = std::move(step2ClassName);
    runsCounter = runIndex;

    beesLogs.clear();
    trajectoriesLogs.clear();
    lostTrajectoriesLogs.clear();
    statsLogs.clear();

    step1InternalTimingLogs.clear();
    step2InternalTimingLogs.clear();
    frameTimingLogs.clear();
    totalTimingLogs.clear();
}

void Logger::incrementRunsCounter() {
    // clear non-performance logs
    beesLogs.clear();
    trajectoriesLogs.clear();
    lostTrajectoriesLogs.clear();
    statsLogs.clear();

    runsCounter++;
}

void Logger::pushStep1InternalTimings(Step1InternalTimings_t &step1InternalTimings) {
    step1Timing.lock();
    step1InternalTimingLogs.emplace_back(runsCounter, step1InternalTimings);
    step1Timing.unlock();
}

void Logger::pushStep2InternalTimings(Step2InternalTimings_t &step2InternalTimings) {
    step2Timing.lock();
    step2InternalTimingLogs.emplace_back(runsCounter, step2InternalTimings);
    step2Timing.unlock();
}

void Logger::pushFrameTimings(FrameTimings_t &frameTimings) {
    frameTiming.lock();
    frameTimingLogs.emplace_back(runsCounter, frameTimings);
    frameTiming.unlock();
}

void Logger::pushTotalTimings(TotalTiming_t& timingTotal) {
    totalTimingLogs.emplace_back(runsCounter, timingTotal);
}

void Logger::writeBenchmarkFiles(std::string& path, std::string filePrefix, bool append, bool step1Internal,
                                 bool step2Internal, bool frames, bool total, bool writeHeader) {
    if (step1Internal) {
        CSVWriter step1InternalFile;
        if ((runsCounter == 0) && writeHeader)
            step1InternalFile.newRow() << "run" << "frameIndex" << "step1Class"
                   << "STORED_FRAMES" << "STEP1_ACCUMULATOR_BLOC_COUNT"
                   << "PATH_DETECTION_RADIUS" << "SINGLE_BEE_DETECTION_RADIUS" << "PATH_MAX_WAIT"
                   << "CORE_COUNT" << "MAX_THREADS_PER_FRAME" << "MAX_FRAMES_IN_PARALLEL"
                   << "T1Timing" << "T2Timing" << "T3Timing" << "T4Timing" << "T5Timing" << "T6Timing"
                   << "extractionTiming" << "filteringTiming" << "waitingTiming" << "step1Timing";

        for (auto& log : step1InternalTimingLogs) {
            uint32_t run = log.first;
            Step1InternalTimings_t& t = log.second;

            step1InternalFile.newRow() << run << t.frameIndex << step1Class
                   << Parameters::STORED_FRAMES << Parameters::STEP1_ACCUMULATOR_BLOC_COUNT
                   << Parameters::PATH_DETECTION_RADIUS << Parameters::SINGLE_BEE_DETECTION_RADIUS << Parameters::PATH_MAX_WAIT
                   << Parameters::CORE_COUNT << Parameters::MAX_THREADS_PER_FRAME << Parameters::MAX_FRAMES_IN_PARALLEL
                   << t.T1Timing << t.T2Timing << t.T3Timing << t.T4Timing << t.T5Timing << t.T6Timing
                   << t.extractionTiming << t.filteringTiming << t.waitingTiming << t.step1Timing;
        }

        step1InternalFile.writeToFile(path + "/" + filePrefix + "step1InternalTimings.csv", append);
    }

    if (step2Internal) {
        CSVWriter step2InternalFile;
        if ((runsCounter == 0) && writeHeader)
            step2InternalFile.newRow() << "run" << "frameIndex" << "step2Class"
                   << "STORED_FRAMES" << "STEP1_ACCUMULATOR_BLOC_COUNT"
                   << "PATH_DETECTION_RADIUS" << "SINGLE_BEE_DETECTION_RADIUS" << "PATH_MAX_WAIT"
                   << "CORE_COUNT" << "MAX_THREADS_PER_FRAME" << "MAX_FRAMES_IN_PARALLEL"
                   << "subStep1Timing" << "subStep2Timing" << "subStep3Timing" << "waitingTiming" << "step2Timing" ;

        for (auto& log : step2InternalTimingLogs) {
            uint32_t run = log.first;
            Step2InternalTimings_t& t = log.second;

            step2InternalFile.newRow() << run << t.frameIndex << step2Class
                   << Parameters::STORED_FRAMES << Parameters::STEP1_ACCUMULATOR_BLOC_COUNT
                   << Parameters::PATH_DETECTION_RADIUS << Parameters::SINGLE_BEE_DETECTION_RADIUS << Parameters::PATH_MAX_WAIT
                   << Parameters::CORE_COUNT << Parameters::MAX_THREADS_PER_FRAME << Parameters::MAX_FRAMES_IN_PARALLEL
                   << t.subStep1Timing << t.subStep2Timing << t.subStep3Timing << t.waitingTiming << t.step2Timing;
        }

        step2InternalFile.writeToFile(path + "/" + filePrefix + "step2InternalTimings.csv", append);
    }

    if (frames) {
        CSVWriter frameFile;
        if ((runsCounter == 0) && writeHeader)
            frameFile.newRow() << "run" << "frameIndex" << "step1Class" << "step2Class"
                   << "STORED_FRAMES" << "STEP1_ACCUMULATOR_BLOC_COUNT"
                   << "PATH_DETECTION_RADIUS" << "SINGLE_BEE_DETECTION_RADIUS" << "PATH_MAX_WAIT"
                   << "CORE_COUNT" << "MAX_THREADS_PER_FRAME" << "MAX_FRAMES_IN_PARALLEL"
                   << "step1Timing" << "step2Timing" << "step3Timing" << "frameTiming";

        for (auto& log : frameTimingLogs) {
            uint32_t run = log.first;
            FrameTimings_t& t = log.second;

            frameFile.newRow() << run << t.frameIndex << step1Class << step2Class
                   << Parameters::STORED_FRAMES << Parameters::STEP1_ACCUMULATOR_BLOC_COUNT
                   << Parameters::PATH_DETECTION_RADIUS << Parameters::SINGLE_BEE_DETECTION_RADIUS << Parameters::PATH_MAX_WAIT
                   << Parameters::CORE_COUNT << Parameters::MAX_THREADS_PER_FRAME << Parameters::MAX_FRAMES_IN_PARALLEL
                   << t.step1Timing << t.step2Timing << t.step3Timing << t.frameTiming;
        }

        frameFile.writeToFile(path + "/" + filePrefix + "framesTimings.csv", append);
    }

    if (total) {
        CSVWriter totalFile;
        if ((runsCounter == 0) && writeHeader)
            totalFile.newRow() << "run" << "step1Class" << "step2Class"
                   << "STORED_FRAMES" << "STEP1_ACCUMULATOR_BLOC_COUNT"
                   << "PATH_DETECTION_RADIUS" << "SINGLE_BEE_DETECTION_RADIUS" << "PATH_MAX_WAIT"
                   << "CORE_COUNT" << "MAX_THREADS_PER_FRAME" << "MAX_FRAMES_IN_PARALLEL"
                   << "allCoresTiming" << "realTiming";

        for (auto& log : totalTimingLogs) {
            uint32_t run = log.first;
            TotalTiming_t& t = log.second;

            totalFile.newRow() << run << step1Class  << step2Class
                   << Parameters::STORED_FRAMES << Parameters::STEP1_ACCUMULATOR_BLOC_COUNT
                   << Parameters::PATH_DETECTION_RADIUS << Parameters::SINGLE_BEE_DETECTION_RADIUS << Parameters::PATH_MAX_WAIT
                   << Parameters::CORE_COUNT << Parameters::MAX_THREADS_PER_FRAME << Parameters::MAX_FRAMES_IN_PARALLEL
                   << t.allCoresTiming << t.realTiming;
        }

        totalFile.writeToFile(path + "/" + filePrefix + "totalTimings.csv", append);
    }
}


