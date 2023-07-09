// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "tensor_device_dispatcher.hpp"

// #include <exception>
// #include <sstream>
// #include <utility>

using namespace std;
using namespace nnfusion;
using namespace nnfusion::pass;
using namespace nnfusion::kernels;

bool TensorDeviceDispatcher::run(std::shared_ptr<InterpreterContext> ctx,
                                 std::shared_ptr<TranslationUnit> tu)
{
    auto& p = tu->program;
    for (auto iterator : p)
    {
        for (auto ins : *iterator)
        {
            if (ins->name() == "Memcpy")
                continue;
            auto gnode = ins->getGNode();
            auto n_device_type = (*ins)["DeviceType"].as<NNFusion_DeviceType>();
            auto n_device_id = (*ins)["DeviceID"].as<int>();
            std::vector<std::shared_ptr<descriptor::Tensor>> all_tensors;
            auto& inputs = ins->get_inputs();
            for (size_t i = 0; i < inputs.size(); i++)
            {
                auto tensor = inputs[i];
                set_tensor_device(tensor, n_device_type, n_device_id);
            }
            auto& outputs = ins->get_outputs();
            for (size_t i = 0; i < outputs.size(); i++)
            {
                auto tensor = outputs[i];
                set_tensor_device(tensor, n_device_type, n_device_id);
            }
            auto& tensors = ins->get_internal_tensors();
            for (size_t i = 0; i < tensors.size(); i++)
            {
                auto tensor = tensors[i];
                set_tensor_device(tensor, n_device_type, n_device_id);
            }
        }
    }
    return true;
}

void TensorDeviceDispatcher::set_tensor_device(std::shared_ptr<descriptor::Tensor> tensor,
                                               NNFusion_DeviceType device_type,
                                               int device_id)
{
    NNFUSION_CHECK(device_type != UNKNOWN);
    NNFUSION_CHECK(device_id != -1);
    auto t_device_type = tensor->get_device_type();
    auto t_device_id = tensor->get_device_id();
    if (t_device_type == UNKNOWN)
        tensor->set_device_type(device_type);
    else
        NNFUSION_CHECK(t_device_type == device_type);

    if (t_device_id == -1)
        tensor->set_device_id(device_id);
    else
        NNFUSION_CHECK(t_device_id == device_id);
}