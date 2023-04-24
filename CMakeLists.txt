cmake_minimum_required(VERSION 3.18)
project(exercise5)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
find_package(Doxygen)

execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpmachine OUTPUT_VARIABLE MACHINE_DUMP)
if (MACHINE_DUMP MATCHES "x86_64")
    message("target machine: x86_64")
    set(MACHINE "x86_64")
elseif (MACHINE_DUMP MATCHES "(aarch64|arm?)")
    message("target machine: Arm")
    set(MACHINE "arm")
else ()
    message(FATAL_ERROR "unsupported processor architecture")
endif()

include_directories(${OpenCV_INCLUDE_DIRS} ${JSONCPP_LIBRARIES} ${JSONCPP_LIBRARIES} utils/ implementation/Inc)
file(GLOB SOURCES
        implementation/Inc/Step1Naive.hpp
        implementation/Inc/Step2Sequential.hpp
        implementation/Inc/Step3.hpp
        implementation/Inc/Step2Parallel.hpp
        implementation/Inc/Step1Optimized.hpp
        implementation/Inc/Step1Simplified.hpp
        implementation/Inc/Logger.hpp
        implementation/Inc/WorkManager.hpp
        implementation/Inc/InputCollection.hpp
        implementation/Inc/GlobalConcurrency.hpp
        implementation/Inc/BenchmarkLogger.hpp

        implementation/Src/Step1Naive.cpp
        implementation/Src/Step2Sequential.cpp
        implementation/Src/Step3.cpp
        implementation/Src/Step2Parallel.cpp
        implementation/Src/Step1Optimized.cpp
        implementation/Src/Step1Simplified.cpp
        implementation/Src/Logger.cpp
        implementation/Src/InputCollection.cpp
        implementation/Src/GlobalConcurrency.cpp
        implementation/Src/BenchmarkLogger.cpp)

file(GLOB UTILS implementation/Inc/FrameLoader.hpp utils/CSVWriter.h)

add_executable(example tests/example.cpp ${SOURCES} ${TARGET_SPECIFIC_SRC} ${UTILS})
if (MACHINE MATCHES "x86_64")
    target_compile_options(example PRIVATE -march=native -O3)
    target_compile_definitions(example PRIVATE X86=1 ARM=0)
elseif (MACHINE MATCHES "(aarch64|arm?)")
    target_compile_options(example PRIVATE -mcpu=cortex-a53 -O3)
    target_compile_definitions(example PRIVATE X86=0 ARM=1)
endif()
target_link_libraries(example PRIVATE Threads::Threads OpenMP::OpenMP_CXX ${OpenCV_LIBS} ${JSONCPP_LIBRARIES})

add_executable(benchmarks tests/benchmarks.cpp ${SOURCES} ${TARGET_SPECIFIC_SRC} ${UTILS})
if (MACHINE MATCHES "x86_64")
    target_compile_options(benchmarks PRIVATE -march=native -O3)
    target_compile_definitions(benchmarks PRIVATE X86=1 ARM=0)
elseif (MACHINE MATCHES "(aarch64|arm?)")
    target_compile_options(benchmarks PRIVATE -mcpu=cortex-a53 -O3)
    target_compile_definitions(benchmarks PRIVATE X86=0 ARM=1)
endif()
target_link_libraries(benchmarks PRIVATE Threads::Threads OpenMP::OpenMP_CXX ${OpenCV_LIBS} ${JSONCPP_LIBRARIES})

if (DOXYGEN_FOUND)
    add_custom_target( doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ../documentation/doxygen/config.in
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif (DOXYGEN_FOUND)