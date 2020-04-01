// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/arithmetic_reduction.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Min-reduction operation.
        class Min : public ArithmeticReduction
        {
        public:
            /// \brief Constructs a min-reduction operation.
            ///
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            Min(const nnfusion::AxisSet& reduction_axes);
        };
    }
}
