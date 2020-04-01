// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/arithmetic_reduction.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Product reduction operation.
        ///
        /// Reduces the tensor, eliminating the specified reduction axes by taking the product.
        class Product : public ArithmeticReduction
        {
        public:
            /// \brief Constructs a product reduction operation.
            ///
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            Product(const nnfusion::AxisSet& reduction_axes);
        };
    }
}
