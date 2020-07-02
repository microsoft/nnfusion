// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <vector>

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"

namespace nnfusion
{
    namespace descriptor
    {
        class Tensor;

        namespace layout
        {
            /// \brief The standard strided layout, used for row-major and column-major, their permutations and slices.
            ///
            /// The linearized offset of an index I is dot(I, strides) + offset.
            class DenseTensorLayout : public TensorLayout
            {
            public:
                ~DenseTensorLayout() override {}
                DenseTensorLayout(const Tensor& tensor);

                size_t get_offset() const { return m_offset; }
                virtual size_t get_index_offset(const std::vector<size_t>& indices) override;
                nnfusion::Strides get_strides() const override;
                virtual bool operator==(const TensorLayout& other) const override;

            protected:
                size_t m_offset{0};
            };
        }
    }
}
