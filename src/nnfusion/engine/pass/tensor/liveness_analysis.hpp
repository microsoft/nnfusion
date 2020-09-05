// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

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

        private:
            void set_tensor_group(shared_ptr<descriptor::Tensor> tensor, const std::string& group);
            std::unordered_set<shared_ptr<descriptor::Tensor>> cross_stream;
        };
    }
}