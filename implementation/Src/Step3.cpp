//
// Created by sylvain on 29/01/23.
//

#include "Step3.hpp"

#include <utility>

typedef enum{
    ENTRANCE,
    BORDER,
    NOWHERE
} PointLocation_t;

Step3::Step3(Logger& logger, Input_t& input) : logger(logger), input(input) {}

/**
 * Helper function to determine if a point is within a rectangle
 * @param rectangle rectangle to test
 * @param point point to test
 * @return true if the point is in the rectangle, false otherwise
 */
static inline bool isInRectangle(Rectangle_t rectangle, Point_t point)
{
    if ((point.x >= rectangle.p1.x) && (point.x <= rectangle.p2.x) &&
        (point.y >= rectangle.p1.y) && (point.y <= rectangle.p2.y))
        return true;

    return false;
}

/**
 * Helper function to get the location of a point
 * @param point point to test
 * @param input reference to input
 * @return point location
 */
static inline PointLocation_t getLocation(Point_t point, Input_t& input)
{
    auto &entrance = input.entranceRectangles;
    auto &borders = input.borderRectangles;

    for (Rectangle_t &rectangle : entrance)
        if (isInRectangle(rectangle, point))
            return ENTRANCE;

    for (Rectangle_t &rectangle : borders)
        if (isInRectangle(rectangle, point))
            return BORDER;

    return NOWHERE;
}

void Step3::processLostTrajectories(std::unique_ptr<std::vector<Trajectory_t>> trajectories, uint32_t frameIndex)
{
    OutputStatistics_t outputStats = {0, 0, 0, 0, 0, 0, 0};

    for (Trajectory_t &trajectory : *trajectories) {
        // extract first and last point
        Point_t firstPoint = trajectory.points[0].coordinates;
        Point_t lastPoint = trajectory.points[trajectory.points.size() - 1].coordinates;

        // get location ot these points
        PointLocation_t firstPointLoc = getLocation(firstPoint, input);
        PointLocation_t lastPointLoc = getLocation(lastPoint, input);

        // classify the trajectory
        if (lastPointLoc == ENTRANCE)
            if (firstPointLoc == BORDER)
                outputStats.in++;
            else if (firstPointLoc == ENTRANCE)
                outputStats.flyByIn++;
            else
                outputStats.lostIn++;
        else if (lastPointLoc == BORDER)
            if (firstPointLoc == ENTRANCE)
                outputStats.out++;
            else if (firstPointLoc == BORDER)
                outputStats.flyByOut++;
            else
                outputStats.lostOut++;
        else
            outputStats.lost++;
    }

    // push lost trajectories if needed
    if (Parameters::DEBUG_FLAGS.logTrajectories)
        logger.pushLostTrajectories(std::move(trajectories), frameIndex);

    // push output to logger
    logger.pushOutputStats(outputStats, frameIndex);
}
