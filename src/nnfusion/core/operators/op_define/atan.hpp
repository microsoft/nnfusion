// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise inverse tangent (arctan) operation.
        ///
        class Atan : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs an arctan operation.
            Atan();
        };
    }
}
