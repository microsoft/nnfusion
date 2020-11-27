// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "data_buffer.hpp"

using namespace nnfusion::test;

TEST(common, data_buffer)
{
    fullTestDataBuffer<char>();
    fullTestDataBuffer<int8_t>();
    fullTestDataBuffer<uint8_t>();
    fullTestDataBuffer<int16_t>();
    fullTestDataBuffer<uint16_t>();
    fullTestDataBuffer<int32_t>();
    fullTestDataBuffer<uint32_t>();
    fullTestDataBuffer<int64_t>();
    fullTestDataBuffer<uint64_t>();
    fullTestDataBuffer<half_float::half>();
    fullTestDataBuffer<float>();
    fullTestDataBuffer<double>();
}