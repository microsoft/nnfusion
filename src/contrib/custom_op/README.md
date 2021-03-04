# Adding Custom Operator in NNFusion

## When do you need to add custom operator?
1. There is an operator that NNFusion doesn't support yet;
2. NNFusion supports this operator, but it doesn't have kernel implementation yet;
3. NNFusion supports both the operator and kernel, but user just want to replace the kernel with a better implementation.

## How to add custom operator?
NNFusion currently supports adding a custom operator in two ways: writing an operator in C++ interface or in a json configure file. 

1. Adding custom operator in C++ interface

    User can use the `REGISTER_OP(#OP_NAME)` macro to register a new operator. It exposes following interfaces to customize an operator:
    - Set op attributes: `attr<#TYPE>(KEY, VALUE)`, e.g.,
    ``` 
    .attr<float>("custom_value", 0.001)
    ```
    - Set shape inference fucntion: `infershape(#FUNC)`, e.g.,
    ```
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(1 == gnode->get_input_size());
        auto& shape_0 = gnode->get_input_shape(0);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), shape_0);
    })
    ```
    - Set Antares IR: `antares_ir(#FUNC)`, e.g.,
    ```
    .antares_ir([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        return op::create_code_from_template(
            "@output0@@data_layout@ = @input0@@data_layout@ * @value@",
            {{"data_layout",
              vector_to_string<std::vector<std::string>>(
                  op::create_layout_from_dims(gnode->get_output_shape(0)))},
             {"value", op->localOpConfig.getRoot()["custom_value"]}});
    })
    ```
    - Set CPU kernel: `cpu_kernel(#FUNC)`, e.g.,
    ```
    .cpu_kernel([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op_config = nnfusion::op::lookup_op_config(gnode->get_op_type());
        float value = op_config.get("custom_value");
        return op::create_code_from_template(
            R"(
                for (int i = 0; i < 1024; i++)
                  output0[i] = inpupt0[i] * @value@;
            )",
            {{"value", std::to_string(value)}});
    })
    ```
    - Set CUDA kernel: `cuda_kernel(#FUCN, CONFIG)`, e.g.,
    ```
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
    ```
    The complete example code can be found in: `src/contrib/custom_op/native/custom_test_op.cpp`.
    Note: after you defined your custom operator, you need to re-build the project.

2. Adding custom operator in JSON format

    The C++ interface allows you to implement arbitray logics in the operator. However, there are few cases where user just want to add a simple operator without many customizaitons. 
    In these scenarios, we provide a way to add operator through configuring a JSON file, without the need to re-build NNFusion.
    For example, the below JSON code implement the same  operator as we introduced above:
    ```
        "ops": [
        {
            "op": "CustomOpTest",
            "custom_value": 0.001,
            "custom_bool": false,
            "input_shapes": [[4, 256], [4, 1024]],
            "output_shapes":[[4, 256], [4, 1024]],
            "antares_ir": "output0[M, N] = input0[M, N] * 0.001",
            "cuda_kernel": "{ int index = blockIdx.x*blockDim.x + threadIdx.x; output0[index] = input0[index] * @custom_value@; }",
            "launch_config": [[1, 1, 4], [1, 1, 32]],
            "cpu_kernel": "{ for (int i = 0; i < 1024; i++) output0[i] = input0[i] * @custom_value@; }"
        }
    ]
    ```
    Note:
    * After written a JSON operator, you can just put it under `$NNFUSION_HOME/custom_op/json/`, then NNFusion will automaticlly import and register this opeartor in the compiler. By default, `NNFUSION_HOME` is set as `$HOME/nnfusion`.
    * In JSON config files, you are not allowed to write arbitrary shape inference logics, thus you can only provide a pair of list of shapes. NNFusion will try to match the input shape and set the output shape as the corresponding one in the `output_shapes` list. If there is no input shape matched, we just set the output shape as the same as input shape. 
    * In both antares_ir or kernel strings, you can always use `@key@` to specify the key defined in operator attributes, NNFusion will automatically replace the key with real value.