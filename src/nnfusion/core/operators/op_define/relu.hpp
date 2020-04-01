// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise Relu operation.
        ///
        class Relu : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a Relu operation.
            ///
            /// \param arg Node that produces the input tensor.
            Relu();
        };

        /// \brief Elementwise ReluBackprop operation.
        ///
        class ReluBackprop : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a ReluBackprop operation.
            ///
            /// \param arg Node that produces the relu forward input tensor.
            ReluBackprop();
        };
    }
}
