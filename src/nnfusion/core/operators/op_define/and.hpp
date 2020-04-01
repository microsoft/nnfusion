// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/binary_elementwise_logical.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise logical-and operation.
        ///
        class And : public BinaryElementwiseLogical
        {
        public:
            /// \brief Constructs a logical-and operation.
            ///
            /// \param arg0 Node that produces the first input tensor.<br>
            /// `[d0, ...]`
            /// \param arg1 Node that produces the second input tensor.<br>
            /// `[d0, ...]`
            ///
            /// Output `[d0, ...]`
            ///
            And();

        protected:
            virtual bool is_commutative() override { return true; }
        };
    }
}
