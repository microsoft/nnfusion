// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        class AllReduce : public Op
        {
        public:
            AllReduce();

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
        };
    }
}
