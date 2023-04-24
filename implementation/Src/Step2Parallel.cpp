//
// Created by sylvain on 30/01/23.
//

#include "Step2Parallel.hpp"

#include <cmath>
#include <thread>
#include <utility>

#define BATCH_SIZE 4

Step2Parallel::Step2Parallel(Logger& logger, GlobalConcurrency &concurrencyManager)
: logger(logger), concurrencyManager(concurrencyManager) {}

std::unique_ptr<std::vector<Trajectory_t>>
Step2Parallel::updateBeeTrajectories(std::unique_ptr<std::vector<Bee_t>> bees, uint32_t frameIndex) {
    auto t0 = std::chrono::steady_clock::now();

    concurrencyManager.lockStep2(frameIndex);

    auto t0Bis = std::chrono::steady_clock::now();

    // get shared tracked bees
    std::vector<Trajectory_t>& trackedBees = concurrencyManager.getTrackedBees();

    auto lostBees = std::make_unique<std::vector<Trajectory_t>>();
    std::vector<std::unique_ptr<std::atomic<bool>>> pickedBees;
    for (uint32_t i = 0; i < bees->size(); i++)
        pickedBees.push_back(std::make_unique<std::atomic<bool>>(false));

    float pathRadius = Parameters::PATH_DETECTION_RADIUS;
    float otherRadius = Parameters::SINGLE_BEE_DETECTION_RADIUS;
    Bee_t *beesData = bees->data();
    uint32_t trajectoryCount = trackedBees.size();

    std::atomic<uint32_t> globalIndex = 0;

    const auto work = [&] {
        uint32_t localIndex;
        do {
            uint32_t start = globalIndex.fetch_add(BATCH_SIZE);
            uint32_t stop = ((start + BATCH_SIZE) < trajectoryCount) ? (start + BATCH_SIZE) : trajectoryCount;
            localIndex = stop;

            for (uint32_t i = start; i < stop; i++) {
                Trajectory_t& t = trackedBees[i];
                FloatPoint_t predictedPosition;

                // sub-step 1

                // simple if ONE_POINT
                if ((t.status == ONE_POINT) || ((t.status == WAITING) && (t.points.size() < 2))) {
                    Point_t p = t.points[0].coordinates;
                    predictedPosition = {(float) p.x, (float) p.y};
                } else {
                    // compute Dt
                    uint32_t nPoints = t.points.size();
                    Point_t p1 = t.points[nPoints - 1].coordinates;
                    Point_t p2 = t.points[nPoints - 2].coordinates;
                    FloatPoint_t deltaPrev = {(float) (p1.x - p2.x), (float) (p1.y - p2.y)};
                    float Dt = atanf(deltaPrev.y / deltaPrev.x);

                    // add delta angle
                    float dAngle = t.points[nPoints - 1].angle - t.points[nPoints - 1].angle; // FIXME check for formula correctness
                    float Da = Dt + dAngle;

                    // compute predicted position
                    predictedPosition = {(float) p1.x + deltaPrev.x*cosf(Da), (float) p1.y + deltaPrev.y*sinf(Da)};
                }

                // sub-step 2

                bool fail = true;
                do {
                    int64_t currentBeeIndex = -1;
                    float currentDistanceSquare = 999999999.9; // infinite :)
                    for (uint32_t k = 0; k < bees->size(); k++) {
//                        if (pickedBees[k])
//                            continue;

                        float radius = (t.status == ALIVE) ? pathRadius : otherRadius;
                        float radiusSquare = radius*radius;
                        FloatPoint_t delta = {(float) beesData[k].coordinates.x - predictedPosition.x,
                                              (float) beesData[k].coordinates.y - predictedPosition.y};

                        // eliminate most of the bees in a simple manner
                        if ((delta.x > radius) || (delta.x < -radius) || (delta.y > radius) || (delta.y < -radius))
                            continue;

                        // compute distance
                        float distanceSquare = delta.x*delta.x + delta.y*delta.y;
                        if (distanceSquare > radiusSquare)
                            continue;

                        // update current most probable bee
                        if ((distanceSquare < currentDistanceSquare) && !pickedBees[k]->load()) {
                            currentDistanceSquare = distanceSquare;
                            currentBeeIndex = k;
                        }
                    }

                    // no bee found
                    if (currentBeeIndex < 0) {
                        if (t.status != WAITING) {
                            t.status = WAITING;
                            t.waitStatesCount = 0;
                        }

                        t.waitStatesCount++;

                        if (t.waitStatesCount >= Parameters::PATH_MAX_WAIT)
                            t.status = LOST;

                        break;
                    }

                    // update trajectory
                    bool f = false;
                    if (std::atomic_compare_exchange_strong(pickedBees[currentBeeIndex].get(), &f, true)) {
                        t.points.push_back(beesData[currentBeeIndex]);
                        t.status = ALIVE;
                        fail = false;
                    }
                } while(fail);
            }
        } while (localIndex < trajectoryCount);
    };

    // create threads
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < Parameters::MAX_THREADS_PER_FRAME; i++)
        threads.emplace_back(work);

    // wait until all threads have finished
    for (std::size_t i = 0; i < Parameters::MAX_THREADS_PER_FRAME; i++)
        threads[i].join();

    auto t1 = std::chrono::steady_clock::now();

    // sub-step 3 remains sequential as it only concerns very few elements and do not leverage more cores

    // remove lost bees
    std::vector<Trajectory_t> tmp = std::move(trackedBees);
    trackedBees.clear();
    for (auto & i : tmp) {
        if (i.status == LOST) {
            lostBees->push_back(i);
        } else {
            trackedBees.push_back(i);
        }
    }
    // add new bees
    for (uint32_t i = 0; i < bees->size(); i++) {
        if (pickedBees[i]->load())
            continue;

        // create new trajectory
        trackedBees.push_back({ONE_POINT, 0, {beesData[i]}});
    }

    auto t2 = std::chrono::steady_clock::now();

    // log bees if needed
    if (Parameters::DEBUG_FLAGS.logBees)
        logger.pushBees(std::move(bees), frameIndex);

    // log current trajectories if needed
    if (Parameters::DEBUG_FLAGS.logTrajectories)
        logger.pushCurrentTrajectories(trackedBees, frameIndex);

    concurrencyManager.unlockStep2();

    if (Parameters::ENABLE_PERFORMANCE_LOG) {
        Step2InternalTimings_t t;
        t.subStep1Timing = 0;
        t.subStep2Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0Bis).count()) / 1000;
        t.subStep3Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000;
        t.waitingTiming = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t0Bis - t0).count()) / 1000;
        t.step2Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count()) / 1000;
        logger.pushStep2InternalTimings(t);
    }

    // return lost bees
    return std::move(lostBees);
}
