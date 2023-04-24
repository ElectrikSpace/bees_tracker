//
// Created by sylvain on 30/01/23.
//

#include "Step1Simplified.hpp"

#include <omp.h>
#include <filesystem>
#include <fstream>

#if X86
#include <emmintrin.h>
#include <smmintrin.h>
#elif ARM
#include <arm_neon.h>
#endif

Step1Simplified::Step1Simplified(Logger &logger, FrameLoader &frameLoader, GlobalConcurrency &concurrencyManager) :
    frameLoader(frameLoader),
    logger(logger),
    concurrencyManager(concurrencyManager),
    T2(frameLoader.getFramesResolution().rows * frameLoader.getFramesResolution().cols),
    T2Bis(frameLoader.getFramesResolution().rows * frameLoader.getFramesResolution().cols),
    T6((frameLoader.getFramesResolution().rows * frameLoader.getFramesResolution().cols) >> (2*Parameters::N_REDUCE))
{
    if (Parameters::DEBUG_FLAGS.generateIntermediateImages) {
        if (!std::filesystem::is_directory("intermediate_results"))
            std::filesystem::create_directory("intermediate_results");
        else
            for (const auto& entry : std::filesystem::directory_iterator("intermediate_results"))
                try {std::filesystem::remove_all(entry.path());} catch(std::exception& e){}

        std::string dirs[5] = {"frames_accumulator", "T2", "T6"};
        for (const std::string& dir : dirs) {
            std::string path = "intermediate_results/" + dir;
            std::filesystem::create_directory(path);
        }
    }
}

#if X86
/**
 * helper function to load a 8x8 bloc when doing horizontal convolution
 * @param buffer pointer to the input buffer
 * @param colCount number of column in the buffer
 * @param bloc pointer to the 8 x __m128 bloc
 * @param inputColIndex reference to the current input column index
 * @param inputBlocIndex reference to the input bloc index
 */
static inline void loadBlocHorizontalConv(uint16_t* buffer, uint32_t colCount, __m128i* bloc,
                                          uint32_t& inputColIndex, uint32_t& inputBlocIndex)
{
    // load raw data
    __m128i tmp[8];
    for (uint32_t j = 0; j < 8; j++)
        tmp[j] = _mm_loadu_si128((__m128i *) &buffer[j*colCount + inputColIndex]);

    // 8x8 matrix transpose is needed

    // first transpose stage
    for (uint32_t i = 0; i < 4; i++) {
        bloc[2 * i] = _mm_unpacklo_epi16(tmp[2 * i], tmp[2 * i + 1]);
        bloc[2 * i + 1] = _mm_unpackhi_epi16(tmp[2 * i], tmp[2 * i + 1]);
    }
    // second transpose stage
    for (uint32_t i = 0; i < 2; i++) {
        tmp[4 * i] = _mm_unpacklo_epi32(bloc[4 * i], bloc[4 * i + 2]);
        tmp[4 * i + 1] = _mm_unpackhi_epi32(bloc[4 * i], bloc[4 * i + 2]);
        tmp[4 * i + 2] = _mm_unpacklo_epi32(bloc[4 * i + 1], bloc[4 * i + 3]);
        tmp[4 * i + 3] = _mm_unpackhi_epi32(bloc[4 * i + 1], bloc[4 * i + 3]);
    }
    // third transpose stage
    for (uint32_t i = 0; i < 4; i++) {
        bloc[2 * i] = _mm_unpacklo_epi64(tmp[i], tmp[i + 4]);
        bloc[2 * i + 1] = _mm_unpackhi_epi64(tmp[i], tmp[i + 4]);
    }

    // update index
    inputColIndex += 8;
    inputBlocIndex = 0;
}
#elif ARM
/**
 * helper function to load a 8x8 bloc when doing horizontal convolution
 * @param buffer pointer to the input buffer
 * @param colCount number of column in the buffer
 * @param bloc pointer to the 8 x uint16x8_t bloc
 * @param inputColIndex reference to the current input column index
 * @param inputBlocIndex reference to the input bloc index
 */
static inline void loadBlocHorizontalConv(uint16_t* buffer, uint32_t colCount, uint16x8_t* bloc,
                                          uint32_t& inputColIndex, uint32_t& inputBlocIndex)
{
    // load raw data
    uint16x8_t tmp1[8];
    for (uint32_t j = 0; j < 8; j++) {
        tmp1[j] = vld1q_u16(&buffer[j*colCount + inputColIndex]);
    }

    // first de-interleave stage
    uint32x4_t tmp2[8];
    for (uint32_t i = 0; i < 4; i++) {
        tmp2[2*i] = vreinterpretq_u32_u16(vzip1q_u16(tmp1[2*i], tmp1[2*i + 1]));
        tmp2[2*i + 1] = vreinterpretq_u32_u16(vzip2q_u16(tmp1[2*i], tmp1[2*i + 1]));
    }

    // second de-interleave stage
    uint64x2_t tmp3[8];
    for (uint32_t i = 0; i < 2; i++) {
        tmp3[4*i] = vreinterpretq_u64_u32(vzip1q_u32(tmp2[4*i], tmp2[4*i + 2]));
        tmp3[4*i + 1] = vreinterpretq_u64_u32(vzip2q_u32(tmp2[4*i], tmp2[4*i + 2]));
        tmp3[4*i + 2] = vreinterpretq_u64_u32(vzip1q_u32(tmp2[4*i + 1], tmp2[4*i + 3]));
        tmp3[4*i + 3] = vreinterpretq_u64_u32(vzip2q_u32(tmp2[4*i + 1], tmp2[4*i + 3]));
    }

    // third de-interleave stage
    for (uint32_t i = 0; i < 4; i++) {
        bloc[2*i] = vreinterpretq_u16_u64(vzip1q_u64(tmp3[i], tmp3[i + 4]));
        bloc[2*i + 1] = vreinterpretq_u16_u64(vzip2q_u64(tmp3[i], tmp3[i + 4]));
    }

    // update index
    inputColIndex += 8;
    inputBlocIndex = 0;
}
#else
/**
 * helper function to load a 8x8 bloc when doing horizontal convolution
 * @param buffer pointer to the input buffer
 * @param colCount number of column in the buffer
 * @param bloc pointer to the 8x8 uint16_t bloc
 * @param inputColIndex reference to the current input column index
 * @param inputBlocIndex reference to the input bloc index
 */
static inline void loadBlocHorizontalConv(const uint16_t* buffer, uint32_t colCount, uint16_t* bloc,
                                          uint32_t& inputColIndex, uint32_t& inputBlocIndex)
{
    for (uint32_t i = 0; i < 8; i++)
        for (uint32_t j = 0; j < 8; j++)
            bloc[8*j + i] = buffer[colCount*i + inputColIndex + j];

    // update index
    inputColIndex += 8;
    inputBlocIndex = 0;
}
#endif

#if X86
/**
 * helper function to store a 8x8 bloc when doing horizontal convolution
 * @param bloc pointer to the 8 x __m128 bloc
 * @param colCount number of column in the output buffer
 * @param buffer pointer to the output buffer
 * @param outputColIndex reference to the current output column index
 * @param outputBlocIndex reference to the output bloc index
 */
static inline void storeBlocHorizontalConv(__m128i* bloc, uint32_t colCount, uint16_t* buffer,
                                           uint32_t& outputColIndex, uint32_t& outputBlocIndex)
{
    // 8x8 matrix transpose is needed

    // first transpose stage
    __m128i tmp[8];
    for (uint32_t i = 0; i < 4; i++) {
        tmp[2 * i] = _mm_unpacklo_epi16(bloc[2 * i], bloc[2 * i + 1]);
        tmp[2 * i + 1] = _mm_unpackhi_epi16(bloc[2 * i], bloc[2 * i + 1]);
    }
    // second transpose stage
    for (uint32_t i = 0; i < 2; i++) {
        bloc[4 * i] = _mm_unpacklo_epi32(tmp[4 * i], tmp[4 * i + 2]);
        bloc[4 * i + 1] = _mm_unpackhi_epi32(tmp[4 * i], tmp[4 * i + 2]);
        bloc[4 * i + 2] = _mm_unpacklo_epi32(tmp[4 * i + 1], tmp[4 * i + 3]);
        bloc[4 * i + 3] = _mm_unpackhi_epi32(tmp[4 * i + 1], tmp[4 * i + 3]);
    }
    // third transpose stage
    for (uint32_t i = 0; i < 4; i++) {
        tmp[2 * i] = _mm_unpacklo_epi64(bloc[i], bloc[i + 4]);
        tmp[2 * i + 1] = _mm_unpackhi_epi64(bloc[i], bloc[i + 4]);
    }

    // store result
    if (outputBlocIndex == 8)
        for (uint32_t i = 0; i < 8; i++)
            _mm_storeu_si128((__m128i *) &buffer[i*colCount + outputColIndex], tmp[i]);
    else {
        // last bloc
        for (uint32_t i = 0; i < 8; i++) {
            uint16_t tmpBuffer[8];
            _mm_storeu_si128((__m128i *) tmpBuffer, tmp[i]);
            for (uint32_t j = 0; j < outputBlocIndex; j++)
                buffer[i*colCount + outputColIndex + j] = tmpBuffer[j];
        }
    }

    // update index
    outputColIndex += outputBlocIndex;
    outputBlocIndex = 0;
}
#elif ARM
/**
 * helper function to store a 8x8 bloc when doing horizontal convolution
 * @param bloc pointer to the 8 x uint16x8_t bloc
 * @param colCount number of column in the output buffer
 * @param buffer pointer to the output buffer
 * @param outputColIndex reference to the current output column index
 * @param outputBlocIndex reference to the output bloc index
 */
static inline void storeBlocHorizontalConv(uint16x8_t* bloc, uint32_t colCount, uint16_t* buffer,
                                           uint32_t& outputColIndex, uint32_t& outputBlocIndex)
{
    // 8x8 matrix transpose is needed

    // first interleave stage
    uint32x4_t tmp1[8];
    for (uint32_t i = 0; i < 4; i++) {
        tmp1[2*i] = vreinterpretq_u32_u16(vzip1q_u16(bloc[2*i], bloc[2*i + 1]));
        tmp1[2*i + 1] = vreinterpretq_u32_u16(vzip2q_u16(bloc[2*i], bloc[2*i + 1]));
    }

    // second de-interleave stage
    uint64x2_t tmp2[8];
    for (uint32_t i = 0; i < 2; i++) {
        tmp2[4*i] = vreinterpretq_u64_u32(vzip1q_u32(tmp1[4*i], tmp1[4*i + 2]));
        tmp2[4*i + 1] = vreinterpretq_u64_u32(vzip2q_u32(tmp1[4*i], tmp1[4*i + 2]));
        tmp2[4*i + 2] = vreinterpretq_u64_u32(vzip1q_u32(tmp1[4*i + 1], tmp1[4*i + 3]));
        tmp2[4*i + 3] = vreinterpretq_u64_u32(vzip2q_u32(tmp1[4*i + 1], tmp1[4*i + 3]));
    }

    // third de-interleave stage
    uint16x8_t tmp3[8];
    for (uint32_t i = 0; i < 4; i++) {
        tmp3[2*i] = vreinterpretq_u16_u64(vzip1q_u64(tmp2[i], tmp2[i + 4]));
        tmp3[2*i + 1] = vreinterpretq_u16_u64(vzip2q_u64(tmp2[i], tmp2[i + 4]));
    }

    // store result
    if (outputBlocIndex == 8)
        for (uint32_t i = 0; i < 8; i++)
            vst1q_u16(&buffer[i*colCount + outputColIndex], tmp3[i]);
    else {
        // last bloc
        for (uint32_t i = 0; i < 8; i++) {
            uint16_t tmpBuffer[8];
            vst1q_u16(tmpBuffer, tmp3[i]);
            for (uint32_t j = 0; j < outputBlocIndex; j++)
                buffer[i*colCount + outputColIndex + j] = tmpBuffer[j];
        }
    }

    // update index
    outputColIndex += outputBlocIndex;
    outputBlocIndex = 0;
}
#else
/**
 * helper function to store a 8x8 bloc when doing horizontal convolution
 * @param bloc pointer to the 8x8 uint16_t bloc
 * @param colCount number of column in the output buffer
 * @param buffer pointer to the output buffer
 * @param outputColIndex reference to the current output column index
 * @param outputBlocIndex reference to the output bloc index
 */
static inline void storeBlocHorizontalConv(const uint16_t* bloc, uint32_t colCount, uint16_t* buffer,
                                           uint32_t& outputColIndex, uint32_t& outputBlocIndex)
{
    for (uint32_t i = 0; i < outputBlocIndex; i++)
        for (uint32_t j = 0; j < 8; j++)
            buffer[colCount*j + outputColIndex + i] = bloc[8*i + j];

    // update index
    outputColIndex += outputBlocIndex;
    outputBlocIndex = 0;
}
#endif

std::unique_ptr<std::vector<Bee_t>> Step1Simplified::processFrame(uint32_t frameIndex) {
    omp_set_num_threads((int) Parameters::MAX_THREADS_PER_FRAME);

    std::atomic<float> waitingTime = 0;
    std::chrono::time_point<std::chrono::steady_clock> timePoints[10];

    timePoints[0] = std::chrono::steady_clock::now();

#pragma omp parallel default(none) shared(frameIndex, waitingTime, timePoints)
{
    // -----------------------------------------------------------------------------------------------------------------
    // BEGINNING OF SUB-STEP 1 AND SUB-STEP 2
    // -----------------------------------------------------------------------------------------------------------------

    // performance counter
    float localWaitingTime = 0;

    // setup data pointer
    uint8_t* frame = frameLoader.getFramePtr(frameIndex);
    int16_t* framesAccumulator = concurrencyManager.getFramesAccumulator().data();
    uint8_t* toRemoveFrame = (frameIndex < Parameters::STORED_FRAMES)
                             ? frameLoader.getInitializationFramePtr(frameIndex)
                             : frameLoader.getFramePtr(frameIndex - Parameters::STORED_FRAMES);

    // use log2 of stored frames count to multiply and divide using shifts
    uint8_t storedFramesLog2 = Parameters::STORED_FRAMES_LOG2;

    // setup weights vectors
    int16_t weights[3] = {Parameters::COLORS_WEIGHTS.r, Parameters::COLORS_WEIGHTS.g, Parameters::COLORS_WEIGHTS.b};
#if X86
    __m128i weightR = _mm_set1_epi16(weights[0]);
    __m128i weightG = _mm_set1_epi16(weights[1]);
    __m128i weightB = _mm_set1_epi16(weights[2]);
#elif ARM
    int16x8x3_t vWeights = vld3q_dup_s16(weights);
#endif

#if X86
    // setup masks for de-interleave operations using SSE 8-bit shuffle instructions
    uint8_t z = (1 << 7); // mask value for zero
    uint8_t maskDataRA[16] = {0, z, 3, z, 6, z,  9, z, 12, z, 15, z, z, z, z, z};
    uint8_t maskDataRB[16] = {z, z, z, z, z, z,  z, z,  z, z,  z, z, 2, z, 5, z};
    uint8_t maskDataGA[16] = {1, z, 4, z, 7, z, 10, z, 13, z,  z, z, z, z, z, z};
    uint8_t maskDataGB[16] = {z, z, z, z, z, z,  z, z,  z, z,  0, z, 3, z, 6, z};
    uint8_t maskDataBA[16] = {2, z, 5, z, 8, z, 11, z, 14, z,  z, z, z, z, z, z};
    uint8_t maskDataBB[16] = {z, z, z, z, z, z,  z, z,  z, z,  1, z, 4, z, 7, z};
    __m128i maskRA = _mm_loadu_si128((__m128i *) maskDataRA);
    __m128i maskRB = _mm_loadu_si128((__m128i *) maskDataRB);
    __m128i maskGA = _mm_loadu_si128((__m128i *) maskDataGA);
    __m128i maskGB = _mm_loadu_si128((__m128i *) maskDataGB);
    __m128i maskBA = _mm_loadu_si128((__m128i *) maskDataBA);
    __m128i maskBB = _mm_loadu_si128((__m128i *) maskDataBB);
#endif

    // number of pixels to process
    FrameResolution_t resolution = frameLoader.getFramesResolution();
    uint32_t elementCount = resolution.rows*resolution.cols;
    uint32_t blocSize = concurrencyManager.getFrameAccumulatorBlocSize();

#pragma omp for ordered
    for (uint32_t k = 0; k < GlobalConcurrency::getFrameAccumulatorBlocCount(); k++) {
        // get bloc start and stop, as well as the stop for simd processing
        uint32_t start = k * blocSize;
        uint32_t stop = ((k + 1) * blocSize <= elementCount) ? (k + 1) * blocSize : elementCount;
        uint32_t simdStop = stop - ((stop - start) % 8);

        // lock bloc
        auto t0 = std::chrono::steady_clock::now();
        concurrencyManager.lockAccumulatorBloc(k, frameIndex);
        auto t1 = std::chrono::steady_clock::now();
        localWaitingTime += ((float) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / 1000;

        // simd processing
        for (uint32_t i = start; i < simdStop; i += 8) {
            // get the index for rgb frames
            uint32_t rgbIndex = 3*i;

#if X86
            // load data from frame and split in two vector of 8 16-bits elements
            __m128i inputRawA = _mm_loadu_si128((__m128i *) &frame[rgbIndex]);
            __m128i inputRawB = _mm_loadu_si64((__m128i *) &frame[rgbIndex + 16]);
            __m128i inputR = _mm_or_si128(_mm_shuffle_epi8(inputRawA, maskRA), _mm_shuffle_epi8(inputRawB, maskRB));
            __m128i inputG = _mm_or_si128(_mm_shuffle_epi8(inputRawA, maskGA), _mm_shuffle_epi8(inputRawB, maskGB));
            __m128i inputB = _mm_or_si128(_mm_shuffle_epi8(inputRawA, maskBA), _mm_shuffle_epi8(inputRawB, maskBB));
            __m128i inputGrey;
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                inputGrey = _mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16
                        (inputR, weightR), _mm_mullo_epi16(inputG, weightG)), _mm_mullo_epi16(inputB, weightB));
                inputGrey = _mm_srai_epi16(inputGrey, Parameters::COLORS_WEIGHTS_LOG2);
            }


            // load data from frame to remove and split in two vector of 8 16-bits elements
            __m128i toRemoveRawA = _mm_loadu_si128((__m128i *) &toRemoveFrame[rgbIndex]);
            __m128i toRemoveRawB = _mm_loadu_si64((__m128i *) &toRemoveFrame[rgbIndex + 16]);
            __m128i toRemoveR = _mm_or_si128(_mm_shuffle_epi8(toRemoveRawA, maskRA),
                                             _mm_shuffle_epi8(toRemoveRawB, maskRB));
            __m128i toRemoveG = _mm_or_si128(_mm_shuffle_epi8(toRemoveRawA, maskGA),
                                             _mm_shuffle_epi8(toRemoveRawB, maskGB));
            __m128i toRemoveB = _mm_or_si128(_mm_shuffle_epi8(toRemoveRawA, maskBA),
                                             _mm_shuffle_epi8(toRemoveRawB, maskBB));
            __m128i toRemoveGrey;
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                toRemoveGrey = _mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16
                        (toRemoveR, weightR), _mm_mullo_epi16(toRemoveG, weightG)), _mm_mullo_epi16(toRemoveB, weightB));
                toRemoveGrey = _mm_srai_epi16(toRemoveGrey, Parameters::COLORS_WEIGHTS_LOG2);
            }

            // load data from accumulator
            __m128i accR, accG, accB, accGrey;
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                accGrey = _mm_loadu_si128((__m128i *) &framesAccumulator[i]);
            } else {
                accR = _mm_loadu_si128((__m128i *) &framesAccumulator[rgbIndex]);
                accG = _mm_loadu_si128((__m128i *) &framesAccumulator[rgbIndex + 8]);
                accB = _mm_loadu_si128((__m128i *) &framesAccumulator[rgbIndex + 16]);
            }

            // perform arithmetic
            __m128i T1R, T1G, T1B, T1Grey;
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                T1Grey = _mm_srai_epi16(_mm_sub_epi16
                     (_mm_slli_epi16(inputGrey, storedFramesLog2), accGrey), storedFramesLog2);
            } else {
                T1R = _mm_mullo_epi16(_mm_srai_epi16(_mm_sub_epi16
                     (_mm_slli_epi16(inputR, storedFramesLog2), accR), storedFramesLog2), weightR);
                T1G = _mm_mullo_epi16(_mm_srai_epi16(_mm_sub_epi16
                     (_mm_slli_epi16(inputG, storedFramesLog2), accG), storedFramesLog2), weightG);
                T1B = _mm_mullo_epi16(_mm_srai_epi16(_mm_sub_epi16
                     (_mm_slli_epi16(inputB, storedFramesLog2), accB), storedFramesLog2), weightB);
            }

            // update accumulator
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                accGrey = _mm_sub_epi16(_mm_add_epi16(accGrey, inputGrey), toRemoveGrey);
                _mm_storeu_si128((__m128i *) &framesAccumulator[i], accGrey);
            } else {
                accR = _mm_sub_epi16(_mm_add_epi16(accR, inputR), toRemoveR);
                accG = _mm_sub_epi16(_mm_add_epi16(accG, inputG), toRemoveG);
                accB = _mm_sub_epi16(_mm_add_epi16(accB, inputB), toRemoveB);
                _mm_storeu_si128((__m128i *) &framesAccumulator[rgbIndex], accR);
                _mm_storeu_si128((__m128i *) &framesAccumulator[rgbIndex + 8], accG);
                _mm_storeu_si128((__m128i *) &framesAccumulator[rgbIndex + 16], accB);
            }

            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                // store
                _mm_storeu_si128((__m128i *) &T2[i], _mm_slli_epi16(T1Grey, Parameters::COLORS_WEIGHTS_LOG2));
            } else {
                // assemble the three colors
                __m128i result = _mm_add_epi16(_mm_add_epi16(T1R, T1G), T1B);
                _mm_storeu_si128((__m128i *) &T2[i], result);
            }
#elif ARM
            // load data from frame and split in two vector of 8 16-bits elements
            uint8x8x3_t inputRaw = vld3_u8(&frame[rgbIndex]);
            int16x8_t inputR = vreinterpretq_s16_u16(vmovl_u8(inputRaw.val[0]));
            int16x8_t inputG = vreinterpretq_s16_u16(vmovl_u8(inputRaw.val[1]));
            int16x8_t inputB = vreinterpretq_s16_u16(vmovl_u8(inputRaw.val[2]));
            int16x8_t inputGrey;
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                inputGrey = vaddq_s16(vaddq_s16(vmulq_s16
                        (inputR, vWeights.val[0]), vmulq_s16(inputG, vWeights.val[1])), vmulq_s16(inputB, vWeights.val[2]));
                inputGrey = vshrq_n_s16(inputGrey, Parameters::COLORS_WEIGHTS_LOG2);
            }

            // load data from frame to remove and split in two vector of 8 16-bits elements
            uint8x8x3_t toRemoveRaw = vld3_u8(&toRemoveFrame[rgbIndex]);
            int16x8_t toRemoveR = vreinterpretq_s16_u16(vmovl_u8(toRemoveRaw.val[0]));
            int16x8_t toRemoveG = vreinterpretq_s16_u16(vmovl_u8(toRemoveRaw.val[1]));
            int16x8_t toRemoveB = vreinterpretq_s16_u16(vmovl_u8(toRemoveRaw.val[2]));
            int16x8_t toRemoveGrey;
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                toRemoveGrey = vaddq_s16(vaddq_s16(vmulq_s16
                        (toRemoveR, vWeights.val[0]), vmulq_s16(toRemoveG, vWeights.val[1])), vmulq_s16(toRemoveB, vWeights.val[2]));
                toRemoveGrey = vshrq_n_s16(toRemoveGrey, Parameters::COLORS_WEIGHTS_LOG2);
            }

            // load data from accumulator
            int16x8x3_t acc;
            int16x8_t accGrey;
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                accGrey = vld1q_s16(&framesAccumulator[i]);

            } else {
                acc = vld1q_s16_x3(&framesAccumulator[rgbIndex]);
            }

            // do arithmetic
            int16x8_t T1R, T1G, T1B, T1Grey;
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                T1Grey = vshrq_n_s16(vsubq_s16(vshlq_n_s16(inputGrey, storedFramesLog2), accGrey), storedFramesLog2);
            } else {
                T1R = vmulq_s16(
                        vshrq_n_s16(vsubq_s16(vshlq_n_s16(inputR, storedFramesLog2), acc.val[0]), storedFramesLog2),
                        vWeights.val[0]);
                T1G = vmulq_s16(
                        vshrq_n_s16(vsubq_s16(vshlq_n_s16(inputG, storedFramesLog2), acc.val[1]), storedFramesLog2),
                        vWeights.val[1]);
                T1B = vmulq_s16(
                        vshrq_n_s16(vsubq_s16(vshlq_n_s16(inputB, storedFramesLog2), acc.val[2]), storedFramesLog2),
                        vWeights.val[2]);
            }

            // update accumulator
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                int16x8_t newAcc = vaddq_s16(vsubq_s16(accGrey, toRemoveGrey), inputGrey);
                vst1q_s16(&framesAccumulator[i], newAcc);
            } else {
                int16x8x3_t newAcc = {vaddq_s16(vsubq_s16(acc.val[0], toRemoveR), inputR),
                                      vaddq_s16(vsubq_s16(acc.val[1], toRemoveG), inputG),
                                      vaddq_s16(vsubq_s16(acc.val[2], toRemoveB), inputB)};
                vst1q_s16_x3(&framesAccumulator[rgbIndex], newAcc);
            }

            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                // store
                vst1q_s16(&T2[i], vshlq_n_s16(T1Grey, Parameters::COLORS_WEIGHTS_LOG2));
            } else {
                // assemble the three colors
                vst1q_s16(&T2[i], vaddq_s16(vaddq_s16(T1R, T1G), T1B));
            }
#else
            // load data from frame and split in two vector of 8 16-bits elements
            int16_t inputR[8], inputG[8], inputB[8], inputGrey[8];
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                for (uint32_t j = 0; j < 8; j++) {
                    inputGrey[j] = (int16_t) (frame[rgbIndex + 3*j]*weights[0]
                            + frame[rgbIndex + 3*j + 1]*weights[1]
                            + frame[rgbIndex + 3*j + 2]*weights[2]);
                    inputGrey[j] >>= Parameters::COLORS_WEIGHTS_LOG2;
                }
            } else {
                for (uint32_t j = 0; j < 8; j++) {
                    inputR[j] = frame[rgbIndex + 3*j];
                    inputG[j] = frame[rgbIndex + 3*j + 1];
                    inputB[j] = frame[rgbIndex + 3*j + 2];
                }
            }

            // load data from frame to remove and split in two vector of 8 16-bits elements
            int16_t toRemoveR[8], toRemoveG[8], toRemoveB[8], toRemoveGrey[8];
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                for (uint32_t j = 0; j < 8; j++) {
                    toRemoveGrey[j] = (int16_t) (toRemoveFrame[rgbIndex + 3*j]*weights[0]
                          + toRemoveFrame[rgbIndex + 3*j + 1]*weights[1]
                          + toRemoveFrame[rgbIndex + 3*j + 2]*weights[2]);
                    toRemoveGrey[j] >>= Parameters::COLORS_WEIGHTS_LOG2;
                }
            } else {
                for (uint32_t j = 0; j < 8; j++) {
                    toRemoveR[j] = toRemoveFrame[rgbIndex + 3*j];
                    toRemoveG[j] = toRemoveFrame[rgbIndex + 3*j + 1];
                    toRemoveB[j] = toRemoveFrame[rgbIndex + 3*j + 2];
                }
            }

            // load data from accumulator
            int16_t accR[8], accG[8], accB[8], accGrey[8];
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                for (uint32_t j = 0; j < 8; j++)
                    accGrey[j] = framesAccumulator[i + j];
            } else {
                for (uint32_t j = 0; j < 8; j++) {
                    accR[j] = framesAccumulator[rgbIndex + j];
                    accG[j] = framesAccumulator[rgbIndex + 8 + j];
                    accB[j] = framesAccumulator[rgbIndex + 16 + j];
                }
            }

            // perform arithmetic
            int16_t T1R[8], T1G[8], T1B[8], T1Grey[8];
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                for (uint32_t j = 0; j < 8; j++)
                    T1Grey[j] = (int16_t) (((inputGrey[j] << storedFramesLog2) - accGrey[j]) >> storedFramesLog2);
            } else {
                for (uint32_t j = 0; j < 8; j++) {
                    T1R[j] = (int16_t) (((inputR[j] << storedFramesLog2) - accR[j]) >> storedFramesLog2);
                    T1G[j] = (int16_t) (((inputG[j] << storedFramesLog2) - accG[j]) >> storedFramesLog2);
                    T1B[j] = (int16_t) (((inputB[j] << storedFramesLog2) - accB[j]) >> storedFramesLog2);
                }
            }

            // update accumulator
            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                for (uint32_t j = 0; j < 8; j++)
                    framesAccumulator[i + j] = (int16_t) (accGrey[j] - toRemoveGrey[j] + inputGrey[j]);
            } else {
                for (uint32_t j = 0; j < 8; j++) {
                    framesAccumulator[rgbIndex + j] = (int16_t) (accR[j] - toRemoveR[j] + inputR[j]);
                    framesAccumulator[rgbIndex + 8 + j] = (int16_t) (accG[j] - toRemoveG[j] + inputG[j]);
                    framesAccumulator[rgbIndex + 16 + j] = (int16_t) (accB[j] - toRemoveB[j] + inputB[j]);
                }
            }

            if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
                // store
                for (uint32_t j = 0; j < 8; j++)
                    T2[i + j] = (int16_t) (T1Grey[j] << Parameters::COLORS_WEIGHTS_LOG2);
            } else {
                // assemble the three colors
                for (uint32_t j = 0; j < 8; j++)
                    T2[i + j] = (int16_t) (T1R[j]*weights[0] + T1G[j]*weights[1] + T1B[j]*weights[2]);
            }
#endif
        }

        if (Parameters::USE_GREY_SCALE_FRAMES_ACCUMULATOR) {
            for (uint32_t i = simdStop; i < stop; i++) {
                int32_t value = 0;
                value += frame[3*i]*weights[0];
                value += frame[3*i + 1]*weights[1];
                value += frame[3*i + 2]*weights[2];
                value >>= Parameters::COLORS_WEIGHTS_LOG2;
                int32_t valueToRemove = 0;
                valueToRemove += toRemoveFrame[3*i]*weights[0];
                valueToRemove += toRemoveFrame[3*i + 1]*weights[1];
                valueToRemove += toRemoveFrame[3*i + 2]*weights[2];
                valueToRemove >>= Parameters::COLORS_WEIGHTS_LOG2;
                T2[i] = (int16_t) (((int32_t) ((Parameters::STORED_FRAMES * value) - framesAccumulator[i]) /
                                    Parameters::STORED_FRAMES) << Parameters::COLORS_WEIGHTS_LOG2);
                framesAccumulator[i] = (int16_t) (framesAccumulator[i] + value - valueToRemove);
            }
        } else {
            for (uint32_t i = simdStop; i < stop; i++) {
                int32_t tmp[3];
                uint32_t baseIndex = 3*i;
                for (uint32_t j = 0; j < 3; j++) {
                    tmp[j] = ((((int32_t) Parameters::STORED_FRAMES)*((int32_t) frame[baseIndex + j]))
                              - framesAccumulator[baseIndex + j]) / (int32_t) Parameters::STORED_FRAMES;
                    framesAccumulator[baseIndex + j] = (int16_t) (framesAccumulator[baseIndex + j]
                              + frame[baseIndex + j] - toRemoveFrame[baseIndex + j]);
                }
                T2[i] = (int16_t) (tmp[0]*weights[0] + tmp[1]*weights[0] + tmp[2]*weights[0]);
            }
        }

        // free bloc
        concurrencyManager.unlockAccumulatorBloc(k);
    }

    waitingTime.fetch_add(localWaitingTime);

    // -----------------------------------------------------------------------------------------------------------------
    // END OF SUB-STEP 1 AND SUB-STEP 2
    // -----------------------------------------------------------------------------------------------------------------

// use a barrier to synchronize all threads
#pragma omp barrier
#pragma omp master
{
    timePoints[1] = std::chrono::steady_clock::now();
}

    // -----------------------------------------------------------------------------------------------------------------
    // BEGINNING OF SUB-STEP 3, SUB-STEP 4 AND SUB-STEP 5
    // -----------------------------------------------------------------------------------------------------------------

    // prepare corrected threshold 1 and associated vectors
    Threshold1_t th = Parameters::THRESHOLD_1;
    th.thMax = (int16_t) (((float) th.thMax) * 1.5625); // apply correction factor (25/16)
    th.thMin = (int16_t) (((float) th.thMin) * 1.5625); // apply correction factor (25/16)
#if X86
    __m128i vThMin = _mm_set1_epi16(th.thMin);
    __m128i vThMax = _mm_set1_epi16(th.thMax);
    __m128i vThFactor = _mm_set1_epi16(th.thFactor);
#elif ARM
    int16x8_t vThMin = vdupq_n_s16(th.thMin);
    int16x8_t vThMax = vdupq_n_s16(th.thMax);
    int16x8_t vThFactor = vdupq_n_s16(th.thFactor);
#endif

    // compute the reduced resolution
    FrameResolution_t reducedRes = {resolution.rows / (2*Parameters::N_REDUCE),
                                    resolution.cols / (2*Parameters::N_REDUCE)};

    // 5x5 blur convolution

    // compute horizontal convolution as first step of separable convolution
    uint32_t simdRows = resolution.rows - (resolution.rows % 8);
    uint32_t simdCols = resolution.cols - (resolution.cols % 8);

#pragma omp for
    for (uint32_t x = 0; x < simdRows; x += 8) {
#if X86
        __m128i inputBloc[8];
        __m128i outputBloc[8];
        __m128i previous[4];
#elif ARM
        uint16x8_t inputBloc[8];
        uint16x8_t outputBloc[8];
        int16x8_t previous[4];
#else
        uint16_t inputBloc[8*8];
        uint16_t outputBloc[8*8];
        uint16_t previous[8*4];
#endif
        uint32_t inputColIndex = 0;
        uint32_t outputColIndex = 0;
        uint32_t inputBlocIndex = 8;
        uint32_t outputBlocIndex = 0;

        // load first bloc
        loadBlocHorizontalConv((uint16_t *) &T2[x*resolution.cols], resolution.cols, inputBloc, inputColIndex, inputBlocIndex);

        // setup accumulator and previous values vectors
#if X86
        __m128i accumulator = inputBloc[0];
        for (uint32_t i = 1; i < 4; i++)
            accumulator = _mm_add_epi16(accumulator, inputBloc[i]);
        for (uint32_t i = 0; i < 4; i++)
            previous[i] = inputBloc[i];
        // manage borders by simple copy
        outputBloc[0] = inputBloc[0];
        outputBloc[1] = inputBloc[1];
#elif ARM
        int16x8_t accumulator = vreinterpretq_s16_u16(inputBloc[0]);
        for (uint32_t i = 1; i < 4; i++)
            accumulator = vaddq_s16(accumulator, vreinterpretq_s16_u16(inputBloc[i]));
        for (uint32_t i = 0; i < 4; i++)
            previous[i] = vreinterpretq_s16_u16(inputBloc[i]);
        // manage borders by simple copy
        outputBloc[0] = inputBloc[0];
        outputBloc[1] = inputBloc[1];
#else
        int16_t accumulator[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t j = 0; j < 8; j++) {
            for (uint32_t i = 0; i < 4; i++) {
                accumulator[j] = (int16_t) (accumulator[j] +  inputBloc[8*i + j]);
                previous[8*i + j] = inputBloc[8*i + j];
            }
            // manage borders by simple copy
            outputBloc[8*0 + j] = inputBloc[8*0 + j];
            outputBloc[8*1 + j] = inputBloc[8*1 + j];
        }
#endif
        inputBlocIndex = 4;
        outputBlocIndex = 2;

        for (uint32_t y = 2; y < simdCols; y++) {
            // load input bloc if it's empty
            if (inputBlocIndex >= 8)
                loadBlocHorizontalConv((uint16_t *) &T2[x*resolution.cols], resolution.cols, inputBloc,
                                       inputColIndex, inputBlocIndex);

#if X86
            // get input
            __m128i input = inputBloc[inputBlocIndex++];
            // add to accumulator
            accumulator = _mm_add_epi16(accumulator, input);
            // put result / 4 in output bloc
            outputBloc[outputBlocIndex++] = _mm_srai_epi16(accumulator, 2);
            // update accumulator
            accumulator = _mm_sub_epi16(accumulator, previous[0]);
            // update previous vectors
            for (uint32_t i = 0; i < 3; i++)
                previous[i] = previous[i + 1];
            previous[3] = input;
#elif ARM
            // get input
            int16x8_t input = vreinterpretq_s16_u16(inputBloc[inputBlocIndex++]);
            // add to accumulator
            accumulator = vaddq_s16(accumulator, input);
            // put result
            outputBloc[outputBlocIndex++] = vreinterpretq_u16_s16(vshrq_n_s16(accumulator, 2));
            // update accumulator
            accumulator = vsubq_s16(accumulator, previous[0]);
            // update previous vectors
            for (uint32_t i = 0; i < 3; i++)
                previous[i] = previous[i + 1];
            previous[3] = input;
#else
            for (uint32_t j = 0; j < 8; j++) {
                // get input
                auto input = (int16_t) inputBloc[8*inputBlocIndex + j];
                // add to accumulator
                accumulator[j] = (int16_t) (accumulator[j] + input);
                // put result / 4 in output bloc
                outputBloc[8*outputBlocIndex + j] = (int16_t) (accumulator[j] >> 2);
                // update accumulator
                accumulator[j] = (int16_t) (accumulator[j] - previous[j]);
                // update previous vectors
                for (uint32_t i = 0; i < 3; i++)
                    previous[8*i + j] = previous[8*(i+1) + j];
                previous[8*3 + j] = input;
            }
            inputBlocIndex++;
            outputBlocIndex++;
#endif

            // store output bloc if it's full
            if (outputBlocIndex == 8)
                storeBlocHorizontalConv(outputBloc, resolution.cols, (uint16_t *) &T2Bis[x*resolution.cols],
                                        outputColIndex, outputBlocIndex);
        }

        // store last bloc
        storeBlocHorizontalConv(outputBloc, resolution.cols, (uint16_t *) &T2Bis[x*resolution.cols],
                                outputColIndex, outputBlocIndex);

        // scalar and simple processing for the remaining elements on the row
        for (uint32_t y = simdCols - 2; y < resolution.cols; y++) {
            if (y < resolution.cols - 2) {
                int32_t tmp = T2[x * resolution.cols + y - 2];
                tmp += T2[x * resolution.cols + y - 1];
                tmp += T2[x * resolution.cols + y];
                tmp += T2[x * resolution.cols + y + 1];
                tmp += T2[x * resolution.cols + y + 2];
                T2Bis[x * resolution.cols + y] = (int16_t) (tmp >> 2);
            }
            else
                T2Bis[x * resolution.cols + y] = T2[x * resolution.cols + y];
        }
    }

#pragma omp single
{
    // scalar and simple processing for the remaining rows
    for (uint32_t x = simdRows; x < resolution.rows; x++) {
        for (uint32_t y = 0; y < resolution.cols; y++) {
            if ((y < 2) || (y >= (resolution.cols - 2)))
                T2Bis[x * resolution.cols + y] = T2[x * resolution.cols + y];
            else {
                int32_t tmp = T2[x*resolution.cols + y - 2];
                tmp += T2[x*resolution.cols + y - 1];
                tmp += T2[x*resolution.cols + y];
                tmp += T2[x*resolution.cols + y + 1];
                tmp += T2[x*resolution.cols + y + 2];
                T2Bis[x*resolution.cols + y] = (int16_t) (tmp >> 2);
            }
        }
    }
}


// use a barrier to synchronise all threads
#pragma omp barrier

    // compute vertical convolution as second step of separable convolution
    simdCols = resolution.cols - (resolution.cols % 8);

#pragma omp for
    for (uint32_t y = 0; y < simdCols; y += 8) {
#if X86
        // setup previous values and accumulator vectors
        __m128i previous[4];
        for (uint32_t i = 0; i < 4; i++)
            previous[i] = _mm_loadu_si128((__m128i *) &T2Bis[i * resolution.cols + y]);
        __m128i accumulator = _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(previous[0], previous[1]), previous[2]), previous[3]);

        // a buffer is used as accumulator during the reduction phase
        // add the borders
        __m128i buffer = _mm_setzero_si128();
        for (uint32_t i = 0; i < 2; i++) {
            // apply threshold 1 using masks
            __m128i belowThMax = _mm_cmplt_epi16(previous[i], vThMax);
            __m128i aboveThMin = _mm_cmpgt_epi16(previous[i], vThMin);
            __m128i vMin = _mm_andnot_si128(aboveThMin, _mm_mullo_epi16(previous[i], vThFactor));
            __m128i vMax = _mm_andnot_si128(belowThMax, previous[i]);
            __m128i thResult = _mm_or_si128(vMin, vMax);

            // add to buffer
            buffer = _mm_add_epi16(buffer, thResult);
        }
#elif ARM
        // setup previous values and accumulator vectors
        int16x8_t previous[4];
        for (uint32_t i = 0; i < 4; i++) {
            previous[i] = vld1q_s16(&T2Bis[i * resolution.cols + y]);
        }
        int16x8_t accumulator = vaddq_s16(vaddq_s16(vaddq_s16(previous[0], previous[1]),
                                                    previous[2]), previous[3]);

        // a buffer is used as accumulator during the reduction phase
        // add the borders
        uint16x8_t buffer = vdupq_n_u16(0);
        for (uint32_t i = 0; i < 2; i++) {
            // apply threshold 1 using masks
            uint16x8_t vMin = vandq_u16(vcleq_s16(previous[i], vThMin), vreinterpretq_u16_s16(vmulq_s16(previous[i], vThFactor)));
            uint16x8_t vMax = vandq_u16(vcgeq_s16(previous[i], vThMax), vreinterpretq_u16_s16(previous[i]));
            uint16x8_t thResult = vorrq_u16(vMin, vMax);

            // add to buffer
            buffer = vaddq_u16(buffer, thResult);
        }
#else
        // setup previous values and accumulator vectors
        int16_t previous[8*4];
        int16_t accumulator[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t j = 0; j < 8; j++) {
            for (uint32_t i = 0; i < 4; i++) {
                previous[i * 8 + j] = T2Bis[i * resolution.cols + y + j];
                accumulator[j] = (int16_t) (accumulator[j] + previous[i * 8 + j]);
            }
        }

        // a buffer is used as accumulator during the reduction phase
        // add the borders
        uint16_t buffer[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t i = 0; i < 2; i++) {
            for (uint32_t j = 0; j < 8; j++) {
                // apply threshold 1 and add to buffer
                if (previous[8*i + j] >= th.thMax)
                    buffer[j] += previous[8*i + j];
                else if (previous[8*i + j] <= th.thMin)
                    buffer[j] += previous[8*i + j]*th.thFactor;
            }
        }
#endif
        uint32_t bufferIndex = 2;

        for (uint32_t x = 2; x < resolution.rows - 2; x++) {
#if X86
            // load input
            __m128i input = _mm_loadu_si128((__m128i *) &T2Bis[(x + 2) * resolution.cols + y]);

            // add to accumulator
            accumulator = _mm_add_epi16(accumulator, input);

            // apply threshold 1 using masks
            __m128i thInput = _mm_srai_epi16(accumulator, 2);
            __m128i belowThMax = _mm_cmplt_epi16(thInput, vThMax);
            __m128i aboveThMin = _mm_cmpgt_epi16(thInput, vThMin);
            __m128i vMin = _mm_andnot_si128(aboveThMin, _mm_mullo_epi16(thInput, vThFactor));
            __m128i vMax = _mm_andnot_si128(belowThMax, thInput);
            __m128i thResult = _mm_or_si128(vMin, vMax);

            // reduce and store if buffer is full
            buffer = _mm_add_epi16(buffer, thResult);
            bufferIndex++;
            if (bufferIndex == 4) {
                buffer = _mm_srai_epi16(buffer, 2);
                buffer = _mm_hadd_epi16(buffer, buffer);
                buffer = _mm_hadd_epi16(buffer, buffer);
                buffer = _mm_srai_epi16(buffer, 2);
                _mm_storeu_si32((uint32_t *) &T6[(x >> 2) * reducedRes.cols + (y >> 2)], buffer);
                buffer = _mm_setzero_si128();
                bufferIndex = 0;
            }

            // update accumulator
            accumulator = _mm_sub_epi16(accumulator, previous[0]);

            // update previous values vectors
            for (uint32_t j = 0; j < 3; j++)
                previous[j] = previous[j + 1];
            previous[3] = input;
#elif ARM
            // load input
            int16x8_t input = vld1q_s16(&T2Bis[(x+2)*resolution.cols + y]);

            // add to accumulator
            accumulator = vaddq_s16(accumulator, input);

            // BEGIN THRESHOLDING + REDUCE

            // apply threshold 1 using masks
            int16x8_t thInput = vshrq_n_s16(accumulator, 2);
            uint16x8_t vMin = vandq_u16(vcleq_s16(thInput, vThMin), vreinterpretq_u16_s16(vmulq_s16(thInput, vThFactor)));
            uint16x8_t vMax = vandq_u16(vcgeq_s16(thInput, vThMax), vreinterpretq_u16_s16(thInput));
            uint16x8_t thResult = vorrq_u16(vMin, vMax);

            // reduce and store if buffer is full
            buffer = vaddq_u16(buffer, thResult);
            bufferIndex++;
            if (bufferIndex == 4) {
                buffer = vshrq_n_u16(buffer, 2);
                buffer = vpaddq_u16(buffer, buffer);
                buffer = vpaddq_u16(buffer, buffer);
                buffer = vshrq_n_u16(buffer, 2);
                vst1q_lane_u32((uint32_t *) &T6[(x >> 2)*reducedRes.cols + (y >> 2)], vreinterpretq_u32_u16(buffer), 0);
                buffer = vdupq_n_u16(0);
                bufferIndex = 0;
            }

            // END THRESHOLDING + REDUCE

            // update accumulator
            accumulator = vsubq_s16(accumulator, previous[0]);

            // update previous values vectors
            for (uint32_t j = 0; j < 3; j++)
                previous[j] = previous[j + 1];
            previous[3] = input;
#else
            int16_t input[8];
            for (uint32_t j = 0; j < 8; j++) {
                // load input
                input[j] = T2Bis[(x + 2) * resolution.cols + y + j];

                // add to accumulator
                accumulator[j] = (int16_t) (accumulator[j] + input[j]);

                // apply threshold 1 and add to buffer
                auto thInput = (int16_t) (accumulator[j] >> 2);
                if (thInput >= th.thMax)
                    buffer[j] += thInput;
                else if (thInput <= th.thMin)
                    buffer[j] += thInput*th.thFactor;
            }
            bufferIndex++;
            if (bufferIndex == 4) {
                for (uint32_t j = 0; j < 2; j++) {
                    uint16_t tmp = buffer[4*j] >> 2;
                    tmp += buffer[4*j + 1] >> 2;
                    tmp += buffer[4*j + 2] >> 2;
                    tmp += buffer[4*j + 3] >> 2;
                    tmp >>= 2;
                    T6[(x >> 2) * reducedRes.cols + (y >> 2) + j] = tmp;
                }
                for (uint16_t& j : buffer)
                    j = 0;
                bufferIndex = 0;
            }

            // update accumulator and previous values vectors
            for (uint32_t j = 0; j < 8; j++) {
                accumulator[j] = (int16_t) (accumulator[j] - previous[8*0 + j]);
                for (uint32_t i = 0; i < 3; i++)
                    previous[8*i + j] = previous[8*(i+1) + j];
                previous[8*3 + j] = input[j];
            }
#endif
        }

        // add border
        if (bufferIndex >= 2) {
            for (uint32_t x = resolution.cols - 2; x < resolution.rows; x++) {
#if X86
                // load input
                __m128i input = _mm_loadu_si128((__m128i *) &T2Bis[x * resolution.cols + y]);

                // apply threshold 1 using masks
                __m128i belowThMax = _mm_cmplt_epi16(input, vThMax);
                __m128i aboveThMin = _mm_cmpgt_epi16(input, vThMin);
                __m128i vMin = _mm_andnot_si128(aboveThMin, _mm_mullo_epi16(input, vThFactor));
                __m128i vMax = _mm_andnot_si128(belowThMax, input);
                __m128i thResult = _mm_or_si128(vMin, vMax);

                // add to buffer
                buffer = _mm_add_epi16(buffer, thResult);
                bufferIndex++;
                if (bufferIndex == 4) {
                    buffer = _mm_srai_epi16(buffer, 2);
                    buffer = _mm_hadd_epi16(buffer, buffer);
                    buffer = _mm_hadd_epi16(buffer, buffer);
                    buffer = _mm_srai_epi16(buffer, 2);
                    _mm_storeu_si32((uint32_t *) &T6[(x >> 2) * reducedRes.cols + (y >> 2)], buffer);
                    buffer = _mm_setzero_si128();
                    bufferIndex = 0;
                }
#elif ARM
                // load input
                int16x8_t input = vld1q_s16(&T2Bis[x * resolution.cols + y]);

                // apply threshold 1 using masks
                uint16x8_t vMin = vandq_u16(vcleq_s16(input, vThMin), vreinterpretq_u16_s16(vmulq_s16(input, vThFactor)));
                uint16x8_t vMax = vandq_u16(vcgeq_s16(input, vThMax), vreinterpretq_u16_s16(input));
                uint16x8_t thResult = vorrq_u16(vMin, vMax);

                // add to buffer
                buffer = vaddq_u16(buffer, thResult);
                bufferIndex++;
                if (bufferIndex == 4) {
                    buffer = vshrq_n_u16(buffer, 2);
                    buffer = vpaddq_u16(buffer, buffer);
                    buffer = vpaddq_u16(buffer, buffer);
                    buffer = vshrq_n_u16(buffer, 2);
                    vst1q_lane_u32((uint32_t *) &T6[(x >> 2)*reducedRes.cols + (y >> 2)], vreinterpretq_u32_u16(buffer), 0);
                    buffer = vdupq_n_u16(0);
                    bufferIndex = 0;
                }
#else
                for (uint32_t j = 0; j < 8; j++) {
                    // load input
                    int16_t input = T2Bis[(x + 2) * resolution.cols + y + j];

                    // apply threshold 1 and add to buffer
                    if (input >= th.thMax)
                        buffer[j] += input;
                    else if (input <= th.thMin)
                        buffer[j] += input*th.thFactor;
                }
                bufferIndex++;
                if (bufferIndex == 4) {
                    for (uint32_t j = 0; j < 2; j++) {
                        uint16_t tmp = buffer[4*j] >> 2;
                        tmp += buffer[4*j + 1] >> 2;
                        tmp += buffer[4*j + 2] >> 2;
                        tmp += buffer[4*j + 3] >> 2;
                        tmp >>= 2;
                        T6[(x >> 2) * reducedRes.cols + (y >> 2) + j] = tmp;
                    }
                    for (uint16_t& j : buffer)
                        j = 0;
                    bufferIndex = 0;
                }
#endif
            }
        }

    }

#pragma omp single
{
    // scalar processing for the remaining elements
    uint32_t maxUsedCol = resolution.cols - (resolution.cols % 4);
    for (uint32_t y = simdCols; y < maxUsedCol; y += 4) {
        uint32_t buffer = 0;
        uint32_t bufferIndex = 0;
        for (uint32_t x = 0; x < resolution.rows; x++) {
            if ((x < 2) || (x >= (resolution.rows - 2)))
                buffer += T2Bis[x * resolution.cols + y];
            else {
                int32_t tmp = T2Bis[(x-2)*resolution.cols + y];
                tmp += T2Bis[(x-1)*resolution.cols + y];
                tmp += T2Bis[x*resolution.cols + y];
                tmp += T2Bis[(x+1)*resolution.cols + y];
                tmp += T2Bis[(x+2)*resolution.cols + y];
                tmp >>= 2;

                if (tmp >= th.thMax)
                    buffer += tmp;
                else if (tmp <= th.thMin)
                    buffer += tmp*th.thFactor;

                bufferIndex++;
                if (bufferIndex == 4) {
                    T6[(x >> 2) * reducedRes.cols + (y >> 2)] = buffer >> 2;
                    buffer = 0;
                    bufferIndex = 0;
                }
            }
        }
    }
}

    // -----------------------------------------------------------------------------------------------------------------
    // END OF SUB-STEP 3, SUB-STEP 4 AND SUB-STEP 5
    // -----------------------------------------------------------------------------------------------------------------

// use a barrier to synchronise all threads
#pragma omp barrier
#pragma omp master
{
    timePoints[2] = std::chrono::steady_clock::now();
}

    // -----------------------------------------------------------------------------------------------------------------
    // BEGINNING OF SUB-STEP 6
    // -----------------------------------------------------------------------------------------------------------------

    // reduced resolution is the new resolution
    resolution = reducedRes;

    // 3x3 blur convolution

    // compute horizontal convolution as first step of separable convolution
    simdRows = reducedRes.rows - (reducedRes.rows % 8);
    simdCols = reducedRes.cols - 2 - (reducedRes.cols % 8);

#pragma omp for
    for (uint32_t x = 0; x < simdRows; x += 8) {
#if X86
        __m128i inputBloc[8];
        __m128i outputBloc[8];
        __m128i previous[2];
#elif ARM
        uint16x8_t inputBloc[8];
        uint16x8_t outputBloc[8];
        uint16x8_t previous[2];
#else
        uint16_t inputBloc[8*8];
        uint16_t outputBloc[8*8];
        uint16_t previous[8*2];
#endif
        uint32_t inputColIndex = 0;
        uint32_t outputColIndex = 1;
        uint32_t inputBlocIndex = 8;
        uint32_t outputBlocIndex = 0;

        loadBlocHorizontalConv(&T6[x*resolution.cols], resolution.cols, inputBloc, inputColIndex, inputBlocIndex);

        // setup accumulator and previous values vectors
#if X86
        __m128i accumulator = _mm_add_epi16(inputBloc[0], inputBloc[1]);
        previous[0] = inputBloc[0];
        previous[1] = inputBloc[1];
#elif ARM
        uint16x8_t accumulator = vaddq_u16(inputBloc[0], inputBloc[1]);
        previous[0] = inputBloc[0];
        previous[1] = inputBloc[1];
#else
        uint16_t accumulator[8];
        for (uint32_t j = 0; j < 8; j++) {
            accumulator[j] = inputBloc[8*0 + j] + inputBloc[8*1 + j];
            previous[8*0 + j] = inputBloc[8*0 + j];
            previous[8*1 + j] = inputBloc[8*1 + j];
        }
#endif
        inputBlocIndex = 2;

        for (uint32_t y = 0; y < simdCols; y++) {
            // check if input bloc is empty
            if (inputBlocIndex >= 8)
                loadBlocHorizontalConv(&T6[x*resolution.cols], resolution.cols, inputBloc, inputColIndex, inputBlocIndex);

#if X86
            // get input
            __m128i input = inputBloc[inputBlocIndex++];
            // add to accumulator
            accumulator = _mm_add_epi16(accumulator, input);
            // put result in output bloc
            outputBloc[outputBlocIndex++] = _mm_srai_epi16(accumulator, 1);
            // update accumulator
            accumulator = _mm_sub_epi16(accumulator, previous[0]);
            // update previous vectors
            previous[0] = previous[1];
            previous[1] = input;
#elif ARM
            // get input
            uint16x8_t input = inputBloc[inputBlocIndex++];
            // add to accumulator
            accumulator = vaddq_u16(accumulator, input);
            // put result in output bloc
            outputBloc[outputBlocIndex++] = vshrq_n_u16(accumulator, 1);
            // update accumulator
            accumulator = vsubq_u16(accumulator, previous[0]);
            // update previous vectors
            previous[0] = previous[1];
            previous[1] = input;
#else
            for (uint32_t j = 0; j < 8; j++) {
                // get input
                uint16_t input = inputBloc[8*inputBlocIndex + j];
                // add to accumulator
                accumulator[j] = accumulator[j] + input;
                // put result in output bloc
                outputBloc[8*outputBlocIndex + j] = accumulator[j] >> 1;
                // update accumulator
                accumulator[j] = accumulator[j] - previous[8*0 + j];
                // update previous vectors
                previous[8*0 + j] = previous[8*1 + j];
                previous[8*1 + j] = input;
            }
            inputBlocIndex++;
            outputBlocIndex++;
#endif

            // check if output bloc is full
            if (outputBlocIndex == 8)
                storeBlocHorizontalConv(outputBloc, resolution.cols, &T6[x*resolution.cols], outputColIndex, outputBlocIndex);
        }

        // store last bloc
        storeBlocHorizontalConv(outputBloc, resolution.cols, &T6[x*resolution.cols], outputColIndex, outputBlocIndex);

        // scalar and simple processing for the remaining elements on the row
        for (uint32_t y = simdCols - 1; y < resolution.cols - 1; y++) {
            int32_t tmp = T6[x*resolution.cols + y - 1];
            tmp += T6[x*resolution.cols + y];
            tmp += T6[x*resolution.cols + y + 1];
            T6[x*resolution.cols + y] = (int16_t) (tmp >> 1);
        }
    }

#pragma omp single
{
    // scalar and simple processing for the remaining rows
    for (uint32_t x = simdRows; x < resolution.rows; x++) {
        for (uint32_t y = 1; y < resolution.cols - 1; y++) {
            uint32_t tmp = T6[x*reducedRes.cols + y - 1];
            tmp += T6[x*resolution.cols + y];
            tmp += T6[x*resolution.cols + y + 1];
            T6[x*resolution.cols + y] = (int16_t) (tmp >> 1);
        }
    }
}

    // compute vertical convolution as second step of separable convolution
    simdCols = reducedRes.cols - (reducedRes.cols % 8);

#pragma omp for
    for (uint32_t y = 0; y < simdCols; y += 8) {
#if X86
        __m128i previous[2];
        __m128i accumulator;

        // load previous
        previous[0] = _mm_loadu_si128((__m128i *) &T6[0*reducedRes.cols + y]);
        previous[1] = _mm_loadu_si128((__m128i *) &T6[1*reducedRes.cols + y]);

        // create accumulator
        accumulator = _mm_add_epi16(previous[0], previous[1]);
#elif ARM
        uint16x8_t previous[2];
        uint16x8_t accumulator;

        // load previous
        previous[0] = vld1q_u16(&T6[0*reducedRes.cols + y]);
        previous[1] = vld1q_u16(&T6[1*reducedRes.cols + y]);

        // create accumulator
        accumulator = vaddq_u16(previous[0], previous[1]);
#else
        uint16_t previous[2*8];
        uint16_t accumulator[8];

        // load previous and create accumulator
        for (uint32_t j = 0; j < 8; j++) {
            previous[8*0 + j] = T6[0*reducedRes.cols + y + j];
            previous[8*1 + j] = T6[1*reducedRes.cols + y + j];
            accumulator[j] = previous[8*0 + j] + previous[8*1 + j];
        }
#endif

        for (uint32_t x = 1; x < reducedRes.rows - 1; x++) {
#if X86
            // load input
            __m128i input = _mm_loadu_si128((__m128i *) &T6[(x+1)*reducedRes.cols + y]);
            // add to accumulator
            accumulator = _mm_add_epi16(accumulator, input);
            // store result
            _mm_storeu_si128((__m128i *) &T6[x*reducedRes.cols + y], _mm_srai_epi16(accumulator, 2));
            // update accumulator
            accumulator = _mm_sub_epi16(accumulator, previous[0]);
            // update previous vectors
            previous[0] = previous[1];
            previous[1] = input;
#elif ARM
            // load input
            uint16x8_t input = vld1q_u16(&T6[(x+1)*reducedRes.cols + y]);
            // add to accumulator
            accumulator = vaddq_u16(accumulator, input);
            // store result
            vst1q_u16(&T6[x*reducedRes.cols + y], vshrq_n_u16(accumulator, 2));
            // update accumulator
            accumulator = vsubq_u16(accumulator, previous[0]);
            // update previous vectors
            previous[0] = previous[1];
            previous[1] = input;
#else
            for (uint32_t j = 0; j < 8; j++) {
                // load input
                uint16_t input = T6[(x+1)*reducedRes.cols + y + j];
                // add to accumulator
                accumulator[j] = accumulator[j] + input;
                // store result
                T6[x*reducedRes.cols + y + j] = accumulator[j] >> 2;
                // update accumulator
                accumulator[j] = accumulator[j] - previous[8*0 + j];
                // update previous vectors
                previous[8*0 + j] = previous[8*1 + j];
                previous[8*1 + j] = input;
            }
#endif
        }
    }

#pragma omp single
{
    // scalar processing for the remaining elements
    for (uint32_t y = simdCols; y < resolution.cols - 1; y++) {
        for (uint32_t x = 1; x < resolution.rows - 1; x++) {
            uint32_t tmp = T6[(x-1)*resolution.cols + y];
            tmp += T6[x*resolution.cols + y];
            tmp += T6[(x+1)*resolution.cols + y];
            T6[x*resolution.cols + y] = (int16_t) (tmp >> 2);
        }
    }
}

    // -----------------------------------------------------------------------------------------------------------------
    // END OF SUB-STEP 6
    // -----------------------------------------------------------------------------------------------------------------

#pragma omp barrier
#pragma omp master
{
    timePoints[3] = std::chrono::steady_clock::now();
}

}

    // generate intermediate images if needed
    if (Parameters::DEBUG_FLAGS.generateIntermediateImages)
        generateIntermediateImages(frameIndex);

    // extract bees
    auto bees = subStep7And8();

    timePoints[4] = std::chrono::steady_clock::now();

    // save bee images if needed
    if (Parameters::DEBUG_FLAGS.generateBeeImages)
        generateBeeImages(*bees, frameIndex);

    // log timing if needed
    if (Parameters::ENABLE_PERFORMANCE_LOG) {
        Step1InternalTimings_t t;
        t.frameIndex = frameIndex;
        t.T1Timing = 0;
        t.T2Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(timePoints[1] - timePoints[0]).count()) / 1000;
        t.T3Timing = 0;
        t.T4Timing = 0;
        t.T5Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(timePoints[2] - timePoints[1]).count()) / 1000;
        t.T6Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(timePoints[3] - timePoints[2]).count()) / 1000;
        t.extractionTiming = ((float) std::chrono::duration_cast<std::chrono::microseconds>(timePoints[4] - timePoints[3]).count()) / 1000;
        t.filteringTiming = 0;
        t.step1Timing = ((float) std::chrono::duration_cast<std::chrono::microseconds>(timePoints[4] - timePoints[0]).count()) / 1000;
        t.waitingTiming = waitingTime;
        t.T1Timing -= t.waitingTiming;
        logger.pushStep1InternalTimings(t);
    }

    return std::move(bees);
}

std::unique_ptr<std::vector<Bee_t>> Step1Simplified::subStep7And8() {
    Threshold2_t th = Parameters::THRESHOLD_2;
    th.thresholdPicking = (int16_t) (((float) th.thresholdPicking * 1.5625 * 1.125));
    FrameResolution_t res = frameLoader.getFramesResolution();
    FrameResolution_t reducedRes = {res.rows/(2*Parameters::N_REDUCE), res.cols/(2*Parameters::N_REDUCE)};

    // extract bees
    auto bees = std::make_unique<std::vector<Bee_t>>();
    auto rawBee = std::vector<Point_t>();
    std::vector<Point_t> localStack;
    for (int32_t x = 0; x < reducedRes.rows; x++) {
        for (int32_t y = 0; y < reducedRes.cols; y++) {
            if (T6[x*reducedRes.cols + y] >= th.thresholdPicking) {
                localStack.push_back({(uint32_t) x, (uint32_t) y});
                rawBee.clear();
                while (!localStack.empty()) {
                    Point_t current = localStack.back();
                    localStack.pop_back();
                    rawBee.push_back(current);

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

                uint32_t detectedBees = 0;

                // two bees threshold
                if (rawBee.size() > th.thresholdTwoBees)
                    detectedBees++;

                // one bee threshold
                if (rawBee.size() > th.thresholdOneBee)
                    detectedBees++;

                if  (detectedBees > 0) {
                    // get min and max
                    Point_t* points = rawBee.data();
                    Point_t min = points[0];
                    Point_t max = points[0];
                    Point_t accumulator = points[0];
                    for (uint32_t k = 1; k < rawBee.size(); k++) {
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
                    for (uint32_t k = 0; k < rawBee.size(); k++) {
                        uint32_t xIndex = (points[k].x - min.x)/delta.x;
                        uint32_t yIndex = (points[k].y - min.y)/delta.y;
                        tiles[3*xIndex + yIndex]++;
                    }

                    // get final angle
                    float angle = 0;
                    if ((tiles[3] + tiles[5]) < (tiles[2] + tiles[6]))
                        angle = 45;
                    if ((tiles[2] + tiles[6]) < (tiles[1] + tiles[7]))
                        angle = 90;
                    if ((tiles[1] + tiles[7]) < (tiles[0] + tiles[8]))
                        angle = 135;

                    // add bee one or multiple times
                    uint32_t xIndex = (accumulator.x / (uint32_t) rawBee.size()) * 2 * Parameters::N_REDUCE;
                    uint32_t yIndex = (accumulator.y / (uint32_t) rawBee.size()) * 2 * Parameters::N_REDUCE;
                    for (uint32_t i = 0; i < detectedBees; i++)
                        bees->push_back({{xIndex, yIndex}, angle});
                }
            }
        }
    }

    return std::move(bees);
}

void Step1Simplified::generateBeeImages(std::vector<Bee_t> &bees, uint32_t frameIndex) {
    if (!std::filesystem::is_directory("tmp_bee_images")) {
        std::filesystem::create_directory("tmp_bee_images/");
        std::filesystem::create_directory("tmp_bee_images/original");
        std::filesystem::create_directory("tmp_bee_images/T2");
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
//        cv::Rect crop((int) baseX, (int) baseY, (int) imgSize, (int) imgSize);
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

        // save sub part of T6
        path = "tmp_bee_images/T6/";
        std::ofstream fileT6(path + filename + "_T6.dat", std::ios::out | std::ios::binary);
        std::vector<uint16_t> tmp2(reducedImgSize*reducedImgSize);
        for (uint32_t x = 0; x < reducedImgSize; x++) {
            for (uint32_t y = 0; y < reducedImgSize; y++) {
                tmp2[x*reducedImgSize + y] = T6[(baseX + x)*reducedRes.cols + baseY + y];
            }
        }
        fileT6.write((const char *) &tmp2[0], (long) (sizeof(uint16_t)*tmp2.size()));
        fileT6.close();
    }
}

void Step1Simplified::generateIntermediateImages(uint32_t frameIndex) {
    FrameResolution_t resolution = frameLoader.getFramesResolution();
    FrameResolution_t reducedRes = {resolution.rows / (2*Parameters::N_REDUCE),
                                    resolution.cols / (2*Parameters::N_REDUCE)};

    cv::Mat img((int) resolution.rows,
                (int) resolution.cols,
                CV_8UC1, cv::Scalar(0));
    auto data = (uint8_t *) img.data;

    cv::Mat imgRed((int) reducedRes.rows,
                   (int) reducedRes.cols,
                   CV_8UC1, cv::Scalar(0));
    auto dataRed = (uint8_t *) imgRed.data;

    cv::Mat img2((int) resolution.rows,
                 (int) resolution.cols,
                 CV_8UC3, cv::Scalar(0, 0, 0));
    auto data2 = (uint8_t *) img2.data;

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

    // generate T2
    for (uint32_t i = 0; i < elementCount; i++)
        data[i] = (uint8_t) (abs(T2[i]) / totalWeights);
    filename = "intermediate_results/T2/frame_" + std::to_string(frameIndex) + "_T2.jpg";
    cv::imwrite(filename, img);

    // generate T5
    Pixel_t w = Parameters::COLORS_WEIGHTS;
    for (uint32_t i = 0; i < reducedRes.rows*reducedRes.cols; i++)
        dataRed[i] = (uint8_t) ((T6[i] / (1.125*1.5625)) / (w.r + w.g + w.b));
    filename = "intermediate_results/T6/frame_" + std::to_string(frameIndex) + "_T6.jpg";
    cv::imwrite(filename, imgRed);
}
