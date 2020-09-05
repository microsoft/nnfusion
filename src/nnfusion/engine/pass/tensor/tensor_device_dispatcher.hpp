// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/op.hpp"

namespace nnfusion
{
    namespace pass
    {
        class TensorDeviceDispatcher : public IInterpreterPass
        {
        public:
            bool run(std::shared_ptr<InterpreterContext> ctx,
                     std::shared_ptr<TranslationUnit> tu) override;

        private:
            void set_tensor_device(std::shared_ptr<descriptor::Tensor> tensor,
                                   NNFusion_DeviceType device_type,
                                   int device_id);
        };
    }
}