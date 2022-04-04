// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "cuda_helper.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"

using CodeWriter = nnfusion::codegen::CodeWriter;

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            void test_cudaMemcpyDtoH(CodeWriter& writer,
                                     shared_ptr<nnfusion::descriptor::Tensor> tensor);
            void test_cudaMemcpyHtoD(CodeWriter& writer,
                                     shared_ptr<nnfusion::descriptor::Tensor> tensor);
            void test_cudaMalloc(CodeWriter& writer,
                                 shared_ptr<nnfusion::descriptor::Tensor> tensor);
            vector<float> test_hostData(CodeWriter& writer,
                                        shared_ptr<nnfusion::descriptor::Tensor> tensor);
            vector<float> test_hostData(CodeWriter& writer,
                                        shared_ptr<nnfusion::descriptor::Tensor> tensor,
                                        vector<float>& d);
            void test_compare(CodeWriter& writer, shared_ptr<nnfusion::descriptor::Tensor> tensor);
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
