// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "contrib/custom_op/custom_op.h"

REGISTER_OP(CustomTestOp)
    .attr<float>("custom_value", 0.001)
    .attr<bool>("custom_bool", false)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(1 == gnode->get_input_size());
        auto& shape_0 = gnode->get_input_shape(0);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), shape_0);
    })
    .antares_ir([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        return op::create_code_from_template(
            "@output0@@data_layout@ = @input0@@data_layout@ * @value@",
            {{"data_layout",
              vector_to_string<std::vector<std::string>>(
                  op::create_layout_from_dims(gnode->get_output_shape(0)))},
             {"value", op->localOpConfig.getRoot()["custom_value"]}});
    })
    .cuda_kernel(
        [](std::shared_ptr<graph::GNode> gnode) -> std::string {
            auto op_config = nnfusion::op::lookup_op_config(gnode->get_op_type());
            float value = op_config.get("custom_value");
            return op::create_code_from_template(
                R"(
                int index = blockIdx.x*blockDim.x + threadIdx.x;
                output0[index] = inpupt0[index] * @value@;
            )",
                {{"value", std::to_string(value)}});
        },
        std::vector<uint32_t>({1, 1, 4 /*gridDim*/, 1, 1, 32 /*blockDim*/}))
    .cpu_kernel([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op_config = nnfusion::op::lookup_op_config(gnode->get_op_type());
        float value = op_config.get("custom_value");
        return op::create_code_from_template(
            R"(
                for (int i = 0; i < 1024; i++)
                  output0[i] = inpupt0[i] * @value@;
            )",
            {{"value", std::to_string(value)}});
    });
