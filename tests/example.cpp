#include <iostream>

#include "WorkManager.hpp"
#include "Logger.hpp"
#include "InputCollection.hpp"

#include "Step1Naive.hpp"
#include "Step1Optimized.hpp"
#include "Step1Simplified.hpp"
#include "Step2Sequential.hpp"
#include "Step2Parallel.hpp"
#include "Step3.hpp"

#include "Parameters.hpp"

int main() {
    auto input = InputCollection::createInput(1, 1);

    WorkManager<Step1Simplified, Step2Sequential, Step3> manager(input, 50);

    auto logger = std::make_shared<Logger>(input, Step1Naive::getClassName(), Step2Sequential::getClassName());

    float timing = manager.runHybrid(logger);

    std::cout << "elapsed time: " << timing << " ms" << std::endl;

    logger->writeOutputFiles("../eval/output/test", *manager.getFrameLoader());
}
