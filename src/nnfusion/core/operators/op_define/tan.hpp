// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise tangent operation.
        ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                                     |
        /// | ----- | --------------------------------- | ----------------------------------------------- |
        /// | `arg` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and numeric element type. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                          |
        /// | ---------------------- | ------------------------------------------------------------------------------------ |
        /// | \f$N[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \tan(\texttt{arg}[i_1,\dots,i_n])\f$ |
        class Tan : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a tangent operation.
            ///
            Tan();
        };
    }
}
