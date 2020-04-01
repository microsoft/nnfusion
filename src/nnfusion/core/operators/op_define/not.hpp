// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise logical negation operation.
        class Not : public Op
        {
        public:
            /// \brief Constructs a logical negation operation.
            ///
            Not();

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
        };
    }
}
