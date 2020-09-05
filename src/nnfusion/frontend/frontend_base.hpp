//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.
#pragma once

#include <stdint.h>

namespace nnfusion
{
    namespace frontend
    {
        // use stdint.h to align with ngraph element_type
        using int8 = int8_t;
        using int16 = int16_t;
        using int32 = int32_t;
        using int64 = int64_t;

        using uint8 = uint8_t;
        using uint16 = uint16_t;
        using uint32 = uint32_t;
        using uint64 = uint64_t;
    } // namespace frontend
} // namespace nnfusion