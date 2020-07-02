// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stddef.h>
#include <stdio.h>
#include <string>
#include <tuple>
#include <vector>

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            std::pair<uint64_t, uint64_t> idiv_magic_u32(uint64_t max_numerator, uint64_t divisor);
            std::pair<uint64_t, uint64_t> idiv_magic_u64(uint64_t divisor);
            uint32_t idiv_ceil(int n, int d);

            // This is commented out because it increases the compile time.
            // It should be moved to a debug header.
            // template <typename T>
            // void print_gpu_tensor(const void* p, size_t element_count)
            // {
            //     std::vector<T> local(element_count);
            //     size_t size_in_bytes = sizeof(T) * element_count;
            //     cuda_memcpyDtH(local.data(), p, size_in_bytes);
            //     LOG(INFO) << "{" << nnfusion::join(local) << "}";
            // }
        }
    }
}
