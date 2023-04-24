//
// Created by sylvain on 19/02/23.
//

#include <iostream>

#include "WorkManager.hpp"
#include "Logger.hpp"
#include "InputCollection.hpp"
#include "FrameLoader.hpp"

#include "Step1Naive.hpp"
#include "Step1Optimized.hpp"
#include "Step1Simplified.hpp"
#include "Step2Sequential.hpp"
#include "Step2Parallel.hpp"
#include "Step3.hpp"

#include "Parameters.hpp"

#if X86
#define RUNS_PER_CONFIG 5
#define FRAMES_PER_RUN 250
#else
#define RUNS_PER_CONFIG 5
#define FRAMES_PER_RUN 50
#endif

int main()
{
    // get input
    auto input = InputCollection::createInput(1, 1);
    auto halfResInput = InputCollection::createInput(1, 2);
    auto quarterResInput = InputCollection::createInput(1, 4);

    std::cout << "Start benchmarking with :" << std::endl;
    std::cout << "  -> input: " << input->name << std::endl;
    std::cout << "  -> runs per config: " << RUNS_PER_CONFIG << std::endl;
    std::cout << "  -> frames per run: " << FRAMES_PER_RUN << std::endl;

    // create frames loader
    std::cout << "frames loader initialization...";
    auto frameLoader = std::make_shared<FrameLoader>(*input, FRAMES_PER_RUN);
    auto halfResFrameLoader = std::make_shared<FrameLoader>(*halfResInput, FRAMES_PER_RUN);
    auto quarterResFrameLoader = std::make_shared<FrameLoader>(*quarterResInput, FRAMES_PER_RUN);
    std::cout << "done" << std::endl;

    // create logger
    std::cout << "logger initialization...";
    auto logger = std::make_shared<Logger>(input, Step1Naive::getClassName(), Step2Sequential::getClassName());
    std::cout << "done" << std::endl;

    std::cout << "removing previous benchmarking results...";
#if X86
    std::string basePath = "../eval/benchmarks_x86/";
#else
    std::string basePath = "../eval/benchmarks_rpi/";
#endif
    if (std::filesystem::is_directory(basePath)) {
        for (const auto& entry : std::filesystem::directory_iterator(basePath))
            std::filesystem::remove_all(entry.path());
    } else {
        std::filesystem::create_directory(basePath);
    }
    std::cout << "done" << std::endl;

    for (uint32_t run = 0; run < RUNS_PER_CONFIG; run++) {
        std::cout << "RUN NUMBER " << run << std::endl;

#if X86
        uint32_t maxCoreCount = 16;
#else
        uint32_t maxCoreCount = 4;
#endif
        for (uint32_t defaultCoreCount = 1; defaultCoreCount <= maxCoreCount; defaultCoreCount *= 2) {
            Parameters::CORE_COUNT = defaultCoreCount;

            // introduction benchmark
            std::cout << "run introduction benchmark...";
            std::string path = basePath + "complete_benchmark";
            if (!std::filesystem::is_directory(path))
                std::filesystem::create_directory(path);
            if (!std::filesystem::is_directory(path + "/outputs_intro"))
                std::filesystem::create_directory(path + "/outputs_intro");
            logger->reset(input, Step1Naive::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Naive, Step2Sequential, Step3> managerIntro(input, frameLoader);
            float timing = managerIntro.runSimpleMulticore(logger);
            logger->writeBenchmarkFiles(path, "INTRO_", true, true, true, true, true, (defaultCoreCount == 1));
            logger->writeOutputFiles(path + "/outputs_intro", *managerIntro.getFrameLoader());
            std::cout << "done (" << timing << ")" << std::endl;

            // BENCHMARK STEP1
            std::cout << "Step 1 benchmarks:" << std::endl;
            path = basePath + "step1";
            if (!std::filesystem::is_directory(path))
                std::filesystem::create_directory(path);

            // run STEP1 NAIVE
            std::cout << "-> run naive...";
            logger->reset(input, Step1Naive::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Naive, Step2Sequential, Step3> manager1(input, frameLoader);
            timing = manager1.runHybrid(logger);
            logger->writeBenchmarkFiles(path, "NAIVE_", true, true, false, false, false, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;

            // run STEP1 OPTIMIZED
            std::cout << "-> run optimized...";
            logger->reset(input, Step1Optimized::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Optimized, Step2Sequential, Step3> manager2(input, frameLoader);
            timing = manager2.runHybrid(logger);
            logger->writeBenchmarkFiles(path, "OPTIMIZED_", true, true, false, false, false, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;

            // run STEP1 SIMPLIFIED -> the best
            std::cout << "-> run simplified...";
            logger->reset(input, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Simplified, Step2Sequential, Step3> manager3(input, frameLoader);
            timing = manager3.runHybrid(logger);
            logger->writeBenchmarkFiles(path, "SIMPLIFIED_", true, true, false, false, false, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;

            // ---> test with multiple STEP1_ACCUMULATOR_BLOC_COUNT values
            std::cout << "-> accumulator blocs count:" << std::endl;
            uint32_t accumulatorBlocs = Parameters::STEP1_ACCUMULATOR_BLOC_COUNT;
            for (uint32_t blocs = 2; blocs <= 128; blocs *= 2) {
                std::cout << "   -> run with " << blocs << " blocs...";
                Parameters::STEP1_ACCUMULATOR_BLOC_COUNT = blocs;
                logger->reset(input, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
                WorkManager<Step1Simplified, Step2Sequential, Step3> manager(input, frameLoader);
                timing = manager.runHybrid(logger);
                logger->writeBenchmarkFiles(path, "SIMPLIFIED_ACCUMULATOR_BLOCS_", true, true, false, false, false, ((defaultCoreCount == 1)) && (blocs == 2));
                std::cout << "done (" << timing << ")" << std::endl;
            }
            Parameters::STEP1_ACCUMULATOR_BLOC_COUNT = accumulatorBlocs;



            // BENCHMARK STEP2
            std::cout << "Step 2 benchmarks:" << std::endl;
            path = basePath + "step2";
            if (!std::filesystem::is_directory(path))
                std::filesystem::create_directory(path);

            // run STEP2 SEQUENTIAL -> best
            std::cout << "-> run sequential...";
            logger->reset(input, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Simplified, Step2Sequential, Step3> manager4(input, frameLoader);
            timing = manager4.runHybrid(logger);
            logger->writeBenchmarkFiles(path, "SEQUENTIAL_", true, false, true, false, false, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;

            // run STEP2 PARALLEL
            std::cout << "-> run parallel...";
            logger->reset(input, Step1Simplified::getClassName(), Step2Parallel::getClassName(), run);
            WorkManager<Step1Simplified, Step2Parallel, Step3> manager5(input, frameLoader);
            timing = manager5.runHybrid(logger);
            logger->writeBenchmarkFiles(path, "PARALLEL_", true, false, true, false, false, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;

            // ---> test with multiple radius
            std::cout << "Step 2 parameters tweaking:" << std::endl;
            std::cout << "-> detection radius:" << std::endl;
            float r1 = Parameters::PATH_DETECTION_RADIUS;
            float r2 = Parameters::SINGLE_BEE_DETECTION_RADIUS;
            for (uint32_t radius = 100; radius < 1000; radius += 100) {
                std::cout << "   -> run with radius = " << radius << "...";
                Parameters::PATH_DETECTION_RADIUS = (float) radius;
                Parameters::SINGLE_BEE_DETECTION_RADIUS = 2 * Parameters::PATH_DETECTION_RADIUS;
                logger->reset(input, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
                WorkManager<Step1Simplified, Step2Sequential, Step3> manager(input, frameLoader);
                timing = manager.runHybrid(logger);
                logger->writeBenchmarkFiles(path, "PARALLEL_RADIUS_", true, false, true, false, false, ((defaultCoreCount == 1)) && (radius == 100));
                std::cout << "done (" << timing << ")" << std::endl;
            }
            Parameters::PATH_DETECTION_RADIUS = r1;
            Parameters::SINGLE_BEE_DETECTION_RADIUS = r2;


            // BENCHMARK WORK SCHEDULING
            std::cout << "Scheduling benchmarks:" << std::endl;
            path = basePath + "scheduling";
            if (!std::filesystem::is_directory(path))
                std::filesystem::create_directory(path);

            // run all cores for each frame
            std::cout << "-> run all cores for each frame...";
            logger->reset(input, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Simplified, Step2Sequential, Step3> manager6(input, frameLoader);
            timing = manager6.runSimpleMulticore(logger);
            logger->writeBenchmarkFiles(path, "ALL_CORE_EACH_FRAME_", true, false, false, true, true, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;

            // run one frame per core
            std::cout << "-> run one core per frame...";
            logger->reset(input, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Simplified, Step2Sequential, Step3> manager7(input, frameLoader);
            timing = manager7.runOneFramePerCore(logger);
            logger->writeBenchmarkFiles(path, "ONE_CORE_PER_FRAME_", true, false, false, true, true, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;

            // run hybrid -> the best
            std::cout << "-> run hybrid...";
            logger->reset(input, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Simplified, Step2Sequential, Step3> manager8(input, frameLoader);
            timing = manager8.runHybrid(logger);
            logger->writeBenchmarkFiles(path, "HYBRID_", true, false, false, true, true, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;

            // ---> test with multiple core count
            std::cout << "Scheduling parameters tweaking:" << std::endl;
            std::cout << "-> core count:" << std::endl;
            uint32_t cores = Parameters::CORE_COUNT;
            for (uint32_t c = 1; c <= 16; c += (c == 1) ? 1 : 2) {
                std::cout << "   -> run with " << c << " cores...";
                Parameters::CORE_COUNT = c;
                logger->reset(input, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
                WorkManager<Step1Simplified, Step2Sequential, Step3> manager(input, frameLoader);
                timing = manager.runHybrid(logger);
                logger->writeBenchmarkFiles(path, "HYBRID_CORES_", true, false, false, true, true, ((defaultCoreCount == 1)) && (c == 1));
                std::cout << "done (" << timing << ")" << std::endl;
            }
            Parameters::CORE_COUNT = cores;

            // BENCHMARK WITH SMALLER RESOLUTIONS
            std::cout << "Resolutions benchmarks:" << std::endl;
            path = basePath + "resolutions";
            if (!std::filesystem::is_directory(path))
                std::filesystem::create_directory(path);
            // run with full resolution
            std::cout << "-> run with full resolution...";
            logger->reset(input, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Simplified, Step2Parallel, Step3> manager9(input, frameLoader);
            timing = manager9.runHybrid(logger);
            logger->writeBenchmarkFiles(path, "FULL_RES_", true, false, false, true, true, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;
            // run with half resolution
            std::cout << "-> run with half resolution...";
            logger->reset(halfResInput, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Simplified, Step2Parallel, Step3> manager10(halfResInput, halfResFrameLoader);
            timing = manager10.runHybrid(logger);
            logger->writeBenchmarkFiles(path, "HALF_RES_", true, false, false, true, true, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;
            // run with quarter resolution
            std::cout << "-> run with quarter resolution...";
            logger->reset(quarterResInput, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Simplified, Step2Parallel, Step3> manager11(quarterResInput, quarterResFrameLoader);
            timing = manager11.runHybrid(logger);
            logger->writeBenchmarkFiles(path, "QUARTER_RES_", true, false, false, true, true, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;

            // BENCHMARK WITH COLOR VS GREY SCALE FRAME ACCUMULATOR
            std::cout << "Color vs grey scale benchmarks:" << std::endl;
            path = basePath + "grey_scale";
            if (!std::filesystem::is_directory(path))
                std::filesystem::create_directory(path);
            // run with colors
            std::cout << "-> run with colors...";
            logger->reset(input, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Simplified, Step2Parallel, Step3> manager12(input, frameLoader);
            timing = manager12.runHybrid(logger);
            logger->writeBenchmarkFiles(path, "COLORS_", true, false, false, true, true, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;
            // run with grey scale
            Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR = true;
            std::cout << "-> run with grey scale...";
            logger->reset(halfResInput, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Simplified, Step2Parallel, Step3> manager13(input, frameLoader);
            timing = manager13.runHybrid(logger);
            logger->writeBenchmarkFiles(path, "GREY_SCALE_", true, false, false, true, true, (defaultCoreCount == 1));
            std::cout << "done (" << timing << ")" << std::endl;
            Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR = false;

            // final benchmark
            std::cout << "run final benchmark...";
            path = basePath + "complete_benchmark";
            if (!std::filesystem::is_directory(path + "/outputs_final"))
                std::filesystem::create_directory(path + "/outputs_final");
            logger->reset(input, Step1Simplified::getClassName(), Step2Sequential::getClassName(), run);
            WorkManager<Step1Simplified, Step2Sequential, Step3> managerFinal(input, frameLoader);
            timing = managerFinal.runHybrid(logger);
            logger->writeBenchmarkFiles(path, "FINAL_", true, true, true, true, true, (defaultCoreCount == 1));
            logger->writeOutputFiles(path + "/outputs_final", *managerFinal.getFrameLoader());
            std::cout << "done (" << timing << ")" << std::endl;
        }
    }

    std::cout << "END !!!" << std::endl;
}