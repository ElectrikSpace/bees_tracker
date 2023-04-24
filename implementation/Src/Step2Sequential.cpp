//
// Created by sylvain on 29/01/23.
//

#include "Step2Sequential.hpp"

#include <cmath>
#include <utility>

Step2Sequential::Step2Sequential(Logger& logger, GlobalConcurrency &concurrencyManager)
: logger(logger), concurrencyManager(concurrencyManager) {}

std::unique_ptr<std::vector<Trajectory_t>>
Step2Sequential::updateBeeTrajectories(std::unique_ptr<std::vector<Bee_t>> bees, uint32_t frameIndex) {
    auto t0 = std::chrono::steady_clock::now();

    concurrencyManager.lockStep2(frameIndex);

    auto t0Bis = std::chrono::steady_clock::now();

    // get shared tracked bees
    std::vector<Trajectory_t>& trackedBees = concurrencyManager.getTrackedBees();

    // sub-step 1
    std::vector<FloatPoint_t> predictedPositions;
    for (Trajectory_t& t : trackedBees) {
        // simple if ONE_POINT
        if ((t.status == ONE_POINT) || ((t.status == WAITING) && (t.points.size() < 2))) {
            Point_t p = t.points[0].coordinates;
            predictedPositions.push_back({(float) p.x, (float) p.y});
            continue;
        }

        // compute Dt
        uint32_t nPoints = t.points.size();
        Point_t p1 = t.points[nPoints - 1].coordinates;
        Point_t p2 = t.points[nPoints - 2].coordinates;
        FloatPoint_t delta = {(float) (p1.x - p2.x), (float) (p1.y - p2.y)};
        float Dt = atanf(delta.y / delta.x);

        // add delta angle
        float dAngle = t.points[nPoints - 1].angle - t.points[nPoints - 1].angle; // FIXME check for formula correctness
        float Da = Dt + dAngle;

        // compute predicted position
        predictedPositions.push_back({(float) p1.x + delta.x*cosf(Da), (float) p1.y + delta.y*sinf(Da)});
    }

    auto t1 = std::chrono::steady_clock::now();

    // sub-step 2
    std::vector<bool> pickedBees(bees->size(), false);
    float pathRadius = Parameters::PATH_DETECTION_RADIUS;
    float otherRadius = Parameters::SINGLE_BEE_DETECTION_RADIUS;
    Bee_t *beesData = bees->data();
    for (uint32_t k = 0; k < trackedBees.size(); k++) {
        int64_t currentBeeIndex = -1;
        float currentDistanceSquare = 999999999.9; // infinite :)
        float radius = (trackedBees[k].status == ALIVE) ? pathRadius : otherRadius;
        float radiusSquare = radius*radius;
        for (uint32_t i = 0; i < bees->size(); i++) {
            if (pickedBees[i])
                continue;

            Bee_t &bee = beesData[i];
            FloatPoint_t delta = {(float) bee.coordinates.x - predictedPositions[k].x,
                                  (float) bee.coordinates.y - predictedPositions[k].y};

            // eliminate most of the bees in a simple manner
            if ((delta.x > radius) || (delta.x < -radius) || (delta.y > radius) || (delta.y < -radius))
                continue;

            // compute distance
            float distanceSquare = delta.x*delta.x + delta.y*delta.y;
            if (distanceSquare > radiusSquare)
                continue;

            // update current most probable bee
            if (distanceSquare < currentDistanceSquare) {
                currentDistanceSquare = distanceSquare;
                currentBeeIndex = i;
            }
        }

        // no bee found
        if (currentBeeIndex < 0) {
            if (trackedBees[k].status != WAITING) {
                trackedBees[k].status = WAITING;
                trackedBees[k].waitStatesCount = 0;
            }

            trackedBees[k].waitStatesCount++;

            if (trackedBees[k].waitStatesCount >= Parameters::PATH_MAX_WAIT)
                trackedBees[k].status = LOST;

            continue;
        }

        // update trajectory
        pickedBees[currentBeeIndex] = true;
        trackedBees[k].points.push_back(beesData[currentBeeIndex]);
        trackedBees[k].status = ALIVE;
    }

    auto t2 = std::chrono::steady_clock::now();

    // sub-step 3
    auto lostBees = std::make_unique<std::vector<Trajectory_t>>();

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
        if (pickedBees[i])
            continue;

        // create new trajectory
        trackedBees.push_back({ONE_POINT, 0, {beesData[i]}});
    }

    // log bees if needed
    if (Parameters::DEBUG_FLAGS.logBees)
        logger.pushBees(std::move(bees), frameIndex);

    // log current trajectories if needed
    if (Parameters::DEBUG_FLAGS.logTrajectories)
        logger.pushCurrentTrajectories(trackedBees, frameIndex);

    concurrencyManager.unlockStep2();


    if (Parameters::ENABLE_PERFORMANCE_LOG) {
        Step2InternalTimings_t t;
        auto t3 = std::chrono::steady_clock::now();
        t.subStep1Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0Bis).count()) / 1000;
        t.subStep2Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000;
        t.subStep3Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count()) / 1000;
        t.waitingTiming = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t0Bis - t0).count()) / 1000;
        t.step2Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(t3 - t0).count()) / 1000;
        logger.pushStep2InternalTimings(t);
    }

    // return lost bees
    return std::move(lostBees);
}
