#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        class H2D : public Op
        {
        public:
            H2D();
            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
        };
    }
}
