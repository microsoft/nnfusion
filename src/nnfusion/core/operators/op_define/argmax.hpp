// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/index_reduction.hpp"

namespace nnfusion
{
    namespace op
    {
        // \brief Computes minimum index along a specified axis for a given tensor
        class ArgMax : public IndexReduction
        {
        public:
            /// \brief Constructs a ArgMax operation.
            ///
            /// \param axis The axis along which to compute an index for maximum
            /// \param index_element_type produce indices. Currently, only int64 or int32 are supported
            ArgMax(size_t axis, const nnfusion::element::Type& index_element_type)
                : IndexReduction("ArgMax", axis, index_element_type)
            {
            }
        };
    }
}
