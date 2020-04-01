// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/arithmetic_reduction.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Max-reduction operation.
        class Max : public ArithmeticReduction
        {
        public:
            /// \brief Constructs a max-reduction operation.
            ///
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            Max(const nnfusion::AxisSet& reduction_axes);
        };
    }
}
