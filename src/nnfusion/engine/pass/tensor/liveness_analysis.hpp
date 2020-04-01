// Microsoft (c) 2019, NNFUSION TEAM
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/op.hpp"

namespace nnfusion
{
    namespace pass
    {
        class TensorLivenessAnalysis : public IInterpreterPass
        {
        public:
            bool run(std::shared_ptr<InterpreterContext> ctx,
                     std::shared_ptr<TranslationUnit> tu) override;
        };
    }
}