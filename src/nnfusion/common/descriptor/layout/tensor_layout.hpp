// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "nnfusion/common/descriptor/tensor.hpp"
namespace nnfusion
{
    namespace descriptor
    {
        namespace layout
        {
            /// \brief Interface for describing implementations of tensor views.
            ///
            /// Kernel selection will need to pay attention to the layout.
            class TensorLayout
            {
            protected:
                TensorLayout(const nnfusion::descriptor::Tensor& tensor);
                TensorLayout(const TensorLayout&) = delete;
                TensorLayout& operator=(const TensorLayout&) = delete;

            public:
                virtual ~TensorLayout() {}
                /// Extent of this view in buffer.
                ///
                /// When we support non-linear buffers, this will need to be something other than size_t.
                size_t get_size() const;
                virtual size_t get_allocated_size();
                /// Offset of an index; useful for slice implementation.
                ///
                /// With non-linear buffers, this will need to be something other than size_t.
                virtual size_t get_index_offset(const std::vector<size_t>& indices) = 0;

                const nnfusion::element::Type& get_element_type() const;
                const nnfusion::Shape& get_shape() const;
                virtual nnfusion::Strides get_strides() const = 0;
                /// \brief Return true if this and other have the same element interpretation
                virtual bool operator==(const TensorLayout& other) const = 0;
                bool operator!=(const TensorLayout& other) const { return !(*this == other); }
            protected:
                const nnfusion::element::Type m_element_type;
                const nnfusion::Shape m_shape;
            };
        }
    }
}
