// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/index_reduction.hpp"

namespace nnfusion
{
    namespace op
    {
        // \brief Computes minimum index along a specified axis for a given tensor
        class ArgMin : public IndexReduction
        {
        public:
            /// \brief Constructs a ArgMin operation.
            ///
            /// \param axis The axis along which to compute an index for minimum
            /// \param index_element_type produce indices. Currently, only int64 or int32 are supported
            ArgMin(size_t axis, const nnfusion::element::Type& index_element_type)
                : IndexReduction("ArgMin", axis, index_element_type)
            {
            }
        };
    }
}
