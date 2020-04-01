// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/binary_elementwise_logical.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise logical-or operation.
        ///
        class Or : public BinaryElementwiseLogical
        {
        public:
            /// \brief Constructs a logical-or operation.
            ///
            Or();

        protected:
            virtual bool is_commutative() override { return true; }
        };
    }
}
