#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        class D2H : public Op
        {
        public:
            D2H();
            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
        };
    }
}
