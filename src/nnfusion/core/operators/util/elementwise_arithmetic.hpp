// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        // Abstract base class for elementwise arithmetic operations
        class ElementwiseArithmetic : public Op
        {
        protected:
            ElementwiseArithmetic(const std::string& node_type);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
        };
    }
}
