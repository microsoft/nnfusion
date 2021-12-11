//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
            //     NNFUSION_LOG(INFO) << "{" << nnfusion::join(local) << "}";
            // }
        }
    }
}
