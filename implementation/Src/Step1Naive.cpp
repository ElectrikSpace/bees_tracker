//
// Created by sylvain on 29/01/23.
//

#include "Step1Naive.hpp"
#include "Parameters.hpp"

#include <omp.h>
#include <filesystem>
#include <fstream>
#include <iostream>

// Gaussian blur
//constexpr int16_t K1[25] = {1, 4,  7,  4,  1,
//                            4, 16, 26, 16, 4,
//                            7, 26, 41, 26, 7,
//                            4, 16, 26, 16, 4,
//                            1, 4,  7,  4,  1};
//constexpr int32_t K1Divider = 273;
//
//constexpr int16_t K2[9] = {1, 2, 1,
//                           2, 4, 2,
//                           1, 2, 1};
//constexpr int32_t K2Divider = 25;


// simple blur is used here as it provides better accuracy
constexpr int32_t K1[25] = {1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1};
constexpr int32_t K1Divider = 25;

constexpr int32_t K2[9] = {1, 1, 1,
                           1, 1, 1,
                           1, 1, 1};
constexpr int32_t K2Divider = 9;

Step1Naive::Step1Naive(Logger &logger, FrameLoader &frameLoader, GlobalConcurrency &concurrencyManager) :
    frameLoader(frameLoader),
    logger(logger),
    concurrencyManager(concurrencyManager),
    T1(3 * frameLoader.getFramesResolution().rows * frameLoader.getFramesResolution().cols),
    T2(frameLoader.getFramesResolution().rows * frameLoader.getFramesResolution().cols),
    T3(frameLoader.getFramesResolution().rows * frameLoader.getFramesResolution().cols),
    T4(frameLoader.getFramesResolution().rows * frameLoader.getFramesResolution().cols),
    T5((frameLoader.getFramesResolution().rows * frameLoader.getFramesResolution().cols) >> (2 * Parameters::N_REDUCE)),
    T6((frameLoader.getFramesResolution().rows * frameLoader.getFramesResolution().cols) >> (2 * Parameters::N_REDUCE))
{
    if (!std::filesystem::is_directory("intermediate_results"))
        std::filesystem::create_directory("intermediate_results");

    std::string dirs[7] = {"frames_accumulator", "T1", "T2", "T3", "T4", "T5", "T6"};
    for (const std::string& dir : dirs) {
        std::string path = "intermediate_results/" + dir;

        if (!std::filesystem::is_directory(path))
            std::filesystem::create_directory(path);
        else
            for (const auto& entry : std::filesystem::directory_iterator(path))
                std::filesystem::remove_all(entry.path());
    }
}

std::unique_ptr<std::vector<Bee_t>> Step1Naive::processFrame(uint32_t frameIndex)
{
    omp_set_num_threads((int) Parameters::MAX_THREADS_PER_FRAME);

    auto t0 = std::chrono::steady_clock::now();

    float waitingTime = subStep1(frameIndex);

    auto t1 = std::chrono::steady_clock::now();

    if (!Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR)
        subStep2();

    auto t2 = std::chrono::steady_clock::now();

    subStep3();

    auto t3 = std::chrono::steady_clock::now();

    subStep4();

    auto t4 = std::chrono::steady_clock::now();

    subStep5();

    auto t5 = std::chrono::steady_clock::now();

    subStep6();

    // generate intermediate image if needed
    if (Parameters::DEBUG_FLAGS.generateIntermediateImages)
        generateIntermediateImages(frameIndex);

    auto t6 = std::chrono::steady_clock::now();

    auto rawBees = subStep7();

    auto t7 = std::chrono::steady_clock::now();

    auto bees = subStep8(std::move(rawBees));

    auto t8 = std::chrono::steady_clock::now();

    // save bee images if needed
    if (Parameters::DEBUG_FLAGS.generateBeeImages)
        generateBeeImages(*bees, frameIndex);

    // log timing if needed
    if (Parameters::ENABLE_PERFORMANCE_LOG) {
        Step1InternalTimings_t t;
        t.frameIndex = frameIndex;
        t.T1Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / 1000;
        t.T2Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000;
        t.T3Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count()) / 1000;
        t.T4Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()) / 1000;
        t.T5Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count()) / 1000;
        t.T6Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count()) / 1000;
        t.extractionTiming = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t7 - t6).count()) / 1000;
        t.filteringTiming = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count()) / 1000;
        t.step1Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t8 - t0).count()) / 1000;
        t.waitingTiming = waitingTime;
        t.T1Timing -= t.waitingTiming;
        logger.pushStep1InternalTimings(t);
    }

    return std::move(bees);
}

float Step1Naive::subStep1(uint32_t frameIndex)
{
    int16_t* framesAccumulator = concurrencyManager.getFramesAccumulator().data();
    uint8_t* frame = frameLoader.getFramePtr(frameIndex);
    uint8_t* toRemoveFrame = (frameIndex < Parameters::STORED_FRAMES)
                             ? frameLoader.getInitializationFramePtr(frameIndex)
                             : frameLoader.getFramePtr(frameIndex - Parameters::STORED_FRAMES);
    uint32_t elementCount = frameLoader.getFramesResolution().rows * frameLoader.getFramesResolution().cols;
    elementCount = (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) ? elementCount : 3*elementCount;
    uint32_t blocSize = concurrencyManager.getFrameAccumulatorBlocSize();
    std::atomic<float> waitingTime = 0;

    // do subtraction and update frames accumulator
#pragma omp parallel for ordered default(none) shared(frame, elementCount, T1, framesAccumulator, toRemoveFrame, frameIndex, blocSize, waitingTime)
    for (uint32_t j = 0; j < GlobalConcurrency::getFrameAccumulatorBlocCount(); j++) {
        // get bloc start and stop, as well as the stop for simd processing
        uint32_t start = j * blocSize;
        uint32_t stop = (j + 1) * blocSize;
        start = (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) ? start : 3*start;
        stop = (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) ? stop : 3*stop;
        stop = (stop <= elementCount) ? stop : elementCount;

        // lock bloc
        auto t0 = std::chrono::steady_clock::now();
        concurrencyManager.lockAccumulatorBloc(j, frameIndex);
        if (Parameters::ENABLE_PERFORMANCE_LOG) {
            auto t1 = std::chrono::steady_clock::now();
            waitingTime.fetch_add(
                    ((float) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / 1000);
        }

        for (uint32_t i = start; i < stop; i++) {
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                int32_t value = 0;
                value += frame[3*i]*Parameters::COLORS_WEIGHTS.r;
                value += frame[3*i + 1]*Parameters::COLORS_WEIGHTS.g;
                value += frame[3*i + 2]*Parameters::COLORS_WEIGHTS.b;
                value >>= Parameters::COLORS_WEIGHTS_LOG2;
                int32_t valueToRemove = 0;
                valueToRemove += toRemoveFrame[3*i]*Parameters::COLORS_WEIGHTS.r;
                valueToRemove += toRemoveFrame[3*i + 1]*Parameters::COLORS_WEIGHTS.g;
                valueToRemove += toRemoveFrame[3*i + 2]*Parameters::COLORS_WEIGHTS.b;
                valueToRemove >>= Parameters::COLORS_WEIGHTS_LOG2;
                T2[i] = (int16_t) (((int32_t) ((Parameters::STORED_FRAMES * value) - framesAccumulator[i]) /
                                   Parameters::STORED_FRAMES) << Parameters::COLORS_WEIGHTS_LOG2);
                framesAccumulator[i] = (int16_t) (framesAccumulator[i] + value - valueToRemove);
            } else {
                T1[i] = (int16_t) ((int32_t) ((Parameters::STORED_FRAMES * frame[i]) - framesAccumulator[i]) /
                                   Parameters::STORED_FRAMES);
                framesAccumulator[i] = (int16_t) (framesAccumulator[i] + frame[i] - toRemoveFrame[i]);
            }
        }

        concurrencyManager.unlockAccumulatorBloc(j);
    }

    return waitingTime.load();
}

void Step1Naive::subStep2()
{
    Pixel_t weight = Parameters::COLORS_WEIGHTS;
    uint32_t toProcessElements = 3*frameLoader.getFramesResolution().rows*frameLoader.getFramesResolution().cols;

    // assemble the three colors
#pragma omp parallel for default(none) shared(weight, toProcessElements, T1, T2)
    for (uint32_t i = 0; i < toProcessElements/3; i++)
        T2[i] = (int16_t) (T1[3*i]*weight.r + T1[3*i + 1]*weight.g + T1[3*i + 2]*weight.b);
}

void Step1Naive::subStep3()
{
    FrameResolution_t resolution = frameLoader.getFramesResolution();

    // 5x5 gaussian blur convolution
#pragma omp parallel for default(none) shared(resolution, K1, T2, T3)
    for (uint32_t x = 0; x < resolution.rows; x++) {
        for (uint32_t y = 0; y < resolution.cols; y++) {
            int32_t tmp = 0;
            for (int32_t i = -2; i < 3; i++) {
                for (int32_t j = -2; j < 3; j++) {
                    if (((x+i) >= 0) && ((x+i) < resolution.rows) && ((y+j) >= 0) && ((y+j) < resolution.cols))
                        tmp += T2[(x+i)*resolution.cols + y + j]*K1[(2+i)*5 + 2+j];
                    else
                        tmp += T2[x*resolution.cols + y];
                }
            }
            T3[x*resolution.cols + y] = (int16_t) (tmp / K1Divider);
        }
    }
}

void Step1Naive::subStep4()
{
    uint32_t toProcessElements = frameLoader.getFramesResolution().rows*frameLoader.getFramesResolution().cols;
    Threshold1_t th = Parameters::THRESHOLD_1;

    // thresholding
#pragma omp parallel for default(none) shared(th, toProcessElements, T3, T4)
    for (uint32_t i = 0; i < toProcessElements; i++) {
        if (T3[i] >= th.thMax)
            T4[i] = T3[i];
        else if (T3[i] <= th.thMin)
            T4[i] = T3[i]*th.thFactor;
        else
            T4[i] = 0;
    }
}

void Step1Naive::subStep5()
{
    FrameResolution_t res = frameLoader.getFramesResolution();
    FrameResolution_t reducedRes = {res.rows/(2*Parameters::N_REDUCE), res.cols/(2*Parameters::N_REDUCE)};

    // reduce
#pragma omp parallel for default(none) shared(reducedRes, T4, T5, res)
    for (uint32_t x = 0; x < reducedRes.rows; x++) {
        for (uint32_t y = 0; y < reducedRes.cols; y++) {
            uint32_t acc = 0;
            uint32_t nRed = Parameters::N_REDUCE;
            for (uint32_t i = 0; i < 2*nRed; i++)
                for (uint32_t j = 0; j < 2*nRed; j++)
                    acc += T4[(2*nRed*x + i)*res.cols + 2*y*nRed + j];
            T5[x*reducedRes.cols + y] = acc >> (2*nRed);
        }
    }
}

void Step1Naive::subStep6()
{
    FrameResolution_t res = frameLoader.getFramesResolution();
    FrameResolution_t reducedRes = {res.rows/(2*Parameters::N_REDUCE), res.cols/(2*Parameters::N_REDUCE)};

    // 3x3 gaussian blur convolution
#pragma omp parallel for default(none) shared(reducedRes, K2, T5, T6)
    for (uint32_t x = 0; x < reducedRes.rows; x++) {
        for (uint32_t y = 0; y < reducedRes.cols; y++) {
            int32_t tmp = 0;
            for (int32_t i = -1; i < 2; i++) {
                for (int32_t j = -1; j < 2; j++) {
                    if ((x+i > 0) && (x+i < reducedRes.rows) && (y+j > 0) && (y+j < reducedRes.cols))
                        tmp += T5[(x+i)*reducedRes.cols + y + j]*K2[(1+i)*3 + 1+j];
                    else
                        tmp += T5[x*reducedRes.cols + y];
                }
            }
            T6[x*reducedRes.cols + y] = (int16_t) (tmp / K2Divider);
        }
    }
}

std::unique_ptr<std::vector<std::unique_ptr<std::vector<Point_t>>>> Step1Naive::subStep7()
{
    Threshold2_t th = Parameters::THRESHOLD_2;
    FrameResolution_t res = frameLoader.getFramesResolution();
    FrameResolution_t reducedRes = {res.rows/(2*Parameters::N_REDUCE), res.cols/(2*Parameters::N_REDUCE)};

    // extract bees
    auto bees = std::make_unique<std::vector<std::unique_ptr<std::vector<Point_t>>>>();
    std::vector<Point_t> localStack;
    for (int32_t x = 0; x < reducedRes.rows; x++) {
        for (int32_t y = 0; y < reducedRes.cols; y++) {
            if (T6[x*reducedRes.cols + y] >= th.thresholdPicking) {
                localStack.push_back({(uint32_t) x, (uint32_t) y});
                auto bee = std::make_unique<std::vector<Point_t>>();
                while (!localStack.empty()) {
                    Point_t current = localStack.back();
                    localStack.pop_back();
                    bee->push_back(current);

                    // check for neighbours
                    auto lx = (int32_t) current.x;
                    auto ly = (int32_t) current.y;
                    if (((lx-1) >= 0) && (T6[(lx-1)*reducedRes.cols + ly] >= th.thresholdPicking)) {
                        localStack.push_back({(uint32_t) lx-1, (uint32_t) ly});
                        T6[(lx-1)*reducedRes.cols + ly] = 0;
                    }
                    if (((ly-1) >= 0) && (T6[lx*reducedRes.cols + ly - 1] >= th.thresholdPicking)) {
                        localStack.push_back({(uint32_t) lx, (uint32_t) ly-1});
                        T6[lx*reducedRes.cols + ly - 1] = 0;
                    }
                    if (((lx+1) < reducedRes.rows) && (T6[(lx+1)*reducedRes.cols + ly] >= th.thresholdPicking)) {
                        localStack.push_back({(uint32_t) lx+1, (uint32_t) ly});
                        T6[(lx+1)*reducedRes.cols + ly] = 0;
                    }
                    if (((ly+1) < reducedRes.cols) && (T6[lx*reducedRes.cols + ly + 1] >= th.thresholdPicking)) {
                        localStack.push_back({(uint32_t) lx, (uint32_t) ly+1});
                        T6[lx*reducedRes.cols + ly + 1] = 0;
                    }
                }

                // two bees threshold
                if (bee->size() > th.thresholdTwoBees) {
                    auto bee2 = std::make_unique<std::vector<Point_t>>();
                    bee2->assign(bee->begin(), bee->end());
                    bees->push_back(std::move(bee2));
                }

                // one bee threshold
                if (bee->size() > th.thresholdOneBee)
                    bees->push_back(std::move(bee));
            }
        }
    }

    return std::move(bees);
}

std::unique_ptr<std::vector<Bee_t>> Step1Naive::subStep8
        (std::unique_ptr<std::vector<std::unique_ptr<std::vector<Point_t>>>> beesPoints)
{
    auto bees = std::make_unique<std::vector<Bee_t>>(beesPoints->size());

    // compute angle and coordinates of each bee
#pragma omp parallel for default(none) shared(beesPoints, bees)
    for (uint32_t i = 0; i < beesPoints->size(); i++){
        auto bee = std::move(beesPoints->at(i));

        // get min and max
        Point_t* points = bee->data();
        Point_t min = points[0];
        Point_t max = points[0];
        Point_t accumulator = points[0];
        for (uint32_t k = 1; k < bee->size(); k++) {
            accumulator.x += points[k].x;
            accumulator.y += points[k].y;
            if (points[k].x < min.x)
                min.x = points[k].x;
            if (points[k].y < min.y)
                min.y = points[k].y;
            if (points[k].x > max.x)
                max.x = points[k].x;
            if (points[k].y > max.y)
                max.y = points[k].y;
        }

        // accumulate points in each tiles
        std::vector<uint32_t> tiles(9, 0);
        Point_t delta = {(max.x - min.x)/3 + 1, (max.y - min.y)/3 + 1};
        for (uint32_t k = 0; k < bee->size(); k++) {
            uint32_t x = (points[k].x - min.x)/delta.x;
            uint32_t y = (points[k].y - min.y)/delta.y;
            tiles[3*x + y]++;
        }

        // get final angle
        float angle = 0;
        if ((tiles[3] + tiles[5]) < (tiles[2] + tiles[6]))
            angle = 45;
        if ((tiles[2] + tiles[6]) < (tiles[1] + tiles[7]))
            angle = 90;
        if ((tiles[1] + tiles[7]) < (tiles[0] + tiles[8]))
            angle = 135;

        // add bee
        uint32_t x = (accumulator.x / (uint32_t) bee->size()) * 2 * Parameters::N_REDUCE;
        uint32_t y = (accumulator.y / (uint32_t) bee->size()) * 2 * Parameters::N_REDUCE;
        (*bees)[i] = {{x, y}, angle};
    }

    return std::move(bees);
}

void Step1Naive::generateBeeImages(std::vector<Bee_t>& bees, uint32_t frameIndex)
{
    if (!std::filesystem::is_directory("tmp_bee_images")) {
        std::filesystem::create_directory("tmp_bee_images/");
        std::filesystem::create_directory("tmp_bee_images/original");
        std::filesystem::create_directory("tmp_bee_images/T2");
        std::filesystem::create_directory("tmp_bee_images/T5");
        std::filesystem::create_directory("tmp_bee_images/T6");
    }

    uint32_t scaleFactor = Parameters::N_REDUCE*2;
    uint32_t imgSize = Parameters::BEE_IMAGE_SIZE;
    uint32_t halfImgSize = imgSize / 2;
    FrameResolution_t res = frameLoader.getFramesResolution();
    for (uint32_t i = 0; i < bees.size(); i++) {
        Bee_t &bee = bees[i];

        int64_t baseX = bee.coordinates.x - halfImgSize;
        int64_t baseY = bee.coordinates.y - halfImgSize;
        baseX = (baseX < 0) ? 0 : ((baseX >= (res.rows - imgSize)) ? (int64) (res.rows - imgSize - 1) : baseX);
        baseY = (baseY < 0) ? 0 : ((baseY >= (res.cols - imgSize)) ? (int64) (res.cols - imgSize - 1) : baseY);

        // save sub part of the original image
        cv::Mat img = frameLoader.getFrameMat(frameIndex)(
                cv::Range((int) baseX, (int) (baseX + imgSize)),
                cv::Range((int) baseY, (int) (baseY + imgSize)));
        std::string path = "tmp_bee_images/original/";
        std::string filename = "f" + std::to_string(frameIndex) + "_bee" + std::to_string(i);
        cv::imwrite(path + filename + ".jpg", img);

        // save sub part of T2
        path = "tmp_bee_images/T2/";
        std::ofstream fileT2(path + filename + "_T2.dat", std::ios::out | std::ios::binary);
        std::vector<int16_t> tmp(imgSize*imgSize);
        for (uint32_t x = 0; x < imgSize; x++) {
            for (uint32_t y = 0; y < imgSize; y++) {
                tmp[x*imgSize + y] = T2[(baseX + x)*res.cols + baseY + y];
            }
        }
        fileT2.write((const char *) &tmp[0], (long) (sizeof(int16_t)*tmp.size()));
        fileT2.close();

        uint32_t reducedImgSize = imgSize / scaleFactor;
        uint32_t reducedHalfImgSize = imgSize / 2;
        FrameResolution_t reducedRes = {res.rows/(2*Parameters::N_REDUCE), res.cols/(2*Parameters::N_REDUCE)};
        baseX = (int64_t) (bee.coordinates.x / scaleFactor) - reducedHalfImgSize;
        baseY = (int64_t) (bee.coordinates.y / scaleFactor) - reducedHalfImgSize;
        baseX = (baseX < 0) ? 0 : ((baseX >= res.rows) ? (int64) (res.rows - reducedHalfImgSize) : baseX);
        baseY = (baseY < 0) ? 0 : ((baseY >= res.cols) ? (int64) (res.cols - reducedHalfImgSize) : baseY);

        // save sub part of T5
        path = "tmp_bee_images/T5/";
        std::ofstream fileT5(path + filename + "_T5.dat", std::ios::out | std::ios::binary);
        std::vector<uint16_t> tmp2(reducedImgSize*reducedImgSize);
        for (uint32_t x = 0; x < reducedImgSize; x++) {
            for (uint32_t y = 0; y < reducedImgSize; y++) {
                tmp2[x*reducedImgSize + y] = T5[(baseX + x)*reducedRes.cols + baseY + y];
            }
        }
        fileT5.write((const char *) &tmp2[0], (long) (sizeof(uint16_t)*tmp2.size()));
        fileT5.close();

        // save sub part of T6
        path = "tmp_bee_images/T6/";
        std::ofstream fileT6(path + filename + "_T6.dat", std::ios::out | std::ios::binary);
        for (uint32_t x = 0; x < reducedImgSize; x++) {
            for (uint32_t y = 0; y < reducedImgSize; y++) {
                tmp2[x*reducedImgSize + y] = T6[(baseX + x)*reducedRes.cols + baseY + y];
            }
        }
        fileT6.write((const char *) &tmp2[0], (long) (sizeof(uint16_t)*tmp2.size()));
        fileT6.close();
    }
}

void Step1Naive::generateIntermediateImages(uint32_t frameIndex)
{
    FrameResolution_t resolution = frameLoader.getFramesResolution();
    FrameResolution_t reducedRes = {resolution.rows / (2*Parameters::N_REDUCE),
                                    resolution.cols / (2*Parameters::N_REDUCE)};

    cv::Mat img((int) resolution.rows,
                (int) resolution.cols,
                CV_8UC1, cv::Scalar(0));
    auto data = (uint8_t *) img.data;

    cv::Mat img2((int) resolution.rows,
                 (int) resolution.cols,
                 CV_8UC3, cv::Scalar(0, 0, 0));
    auto data2 = (uint8_t *) img2.data;

    cv::Mat imgRed((int) reducedRes.rows,
                   (int) reducedRes.cols,
                   CV_8UC1, cv::Scalar(0));
    auto dataRed = (uint8_t *) imgRed.data;

    uint32_t elementCount = frameLoader.getFramesResolution().rows*frameLoader.getFramesResolution().cols;
    int16_t* framesAccumulator = concurrencyManager.getFramesAccumulator().data();
    int16_t totalWeights = Parameters::COLORS_WEIGHTS.r + Parameters::COLORS_WEIGHTS.b + Parameters::COLORS_WEIGHTS.b;

    // generate accumulator
    std::string filename = "intermediate_results/frames_accumulator/frame_" + std::to_string(frameIndex) + "_accumulator.jpg";
    if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
        for (uint32_t i = 0; i < elementCount; i++)
            data[i] = (uint8_t) (framesAccumulator[i] / Parameters::STORED_FRAMES);
        cv::imwrite(filename, img);
    } else {
        for (uint32_t i = 0; i < 3*elementCount; i++)
            data2[i] = (uint8_t) (framesAccumulator[i] / Parameters::STORED_FRAMES);
        cv::imwrite(filename, img2);
    }

    // generate T1
    for (uint32_t i = 0; i < 3*elementCount; i++)
        data2[i] = (uint8_t) (abs(T1[i]));
    filename = "intermediate_results/T1/frame_" + std::to_string(frameIndex) + "_T1.jpg";
    cv::imwrite(filename, img2);

    // generate T2
    for (uint32_t i = 0; i < elementCount; i++)
        data[i] = (uint8_t) (abs(T2[i]) / totalWeights);
    filename = "intermediate_results/T2/frame_" + std::to_string(frameIndex) + "_T2.jpg";
    cv::imwrite(filename, img);

    // generate T3
    Pixel_t w = Parameters::COLORS_WEIGHTS;
    for (uint32_t i = 0; i < resolution.rows*resolution.cols; i++)
        data[i] = (uint8_t) (abs((int16_t) (T3[i]) / (w.r + w.g + w.b)));
    filename = "intermediate_results/T3/frame_" + std::to_string(frameIndex) + "_T3.jpg";
    cv::imwrite(filename, img);

    // generate T4
    for (uint32_t i = 0; i < resolution.rows*resolution.cols; i++)
        data[i] = (uint8_t) (abs((int16_t) (T4[i]) / (w.r + w.g + w.b)));
    filename = "intermediate_results/T4/frame_" + std::to_string(frameIndex) + "_T3.jpg";
    cv::imwrite(filename, img);

    // generate T5
    w = Parameters::COLORS_WEIGHTS;
    for (uint32_t i = 0; i < reducedRes.rows*reducedRes.cols; i++)
        dataRed[i] = (uint8_t) (T5[i] / (w.r + w.g + w.b));
    filename = "intermediate_results/T5/frame_" + std::to_string(frameIndex) + "_T5.jpg";
    cv::imwrite(filename, imgRed);

    // generate T6
    for (uint32_t i = 0; i < reducedRes.rows*reducedRes.cols; i++)
        dataRed[i] = (uint8_t) (T6[i] / (w.r + w.g + w.b));
    filename = "intermediate_results/T6/frame_" + std::to_string(frameIndex) + "_T6.jpg";
    cv::imwrite(filename, imgRed);
}
