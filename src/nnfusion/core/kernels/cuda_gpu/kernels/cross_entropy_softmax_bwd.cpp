// Microsoft (c) 2019, NNFusion Team

// This is the 2rd-generation of kernel definition, recommend to extend new ops with this style
// Changes needed for creating an new kernel with 2rd generation style.
//

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

#define __KernelOpType__ "CrossEntropyFwdBwdWithSoftmaxBwd"
#define __KernelUniqueClassName__ CrossEntropyFwdBwdWithSoftmaxBwd_impl0

namespace
{
    nnfusion::op::OpConfig::any
        generate_kernel_code(const std::vector<nnfusion::Shape>& input_shapes,
                             const std::vector<nnfusion::Shape>& output_shapes,
                             const nnfusion::op::OpConfig::any& config)
    {
        // e.g: inputs_shapes = {{64, 10}, {64}}, output_shapes = {{64, 10}}, config = {}

        if (input_shapes[1][0] > 1024)
            return nullptr; // following kernel implementation not supporting that case

        auto src = nnfusion::op::create_code_from_template(R"(
    output0[blockIdx.x * @step_size@ + threadIdx.x] = input0[blockIdx.x * @step_size@ + threadIdx.x] + ((int)input1[blockIdx.x] == threadIdx.x ? -1 : 0);
                        )",
                                                           {
                                                               {"step_size", input_shapes[0][1]},
                                                           });

        return nnfusion::op::OpConfig::any({
            {"block_dim", {input_shapes[0][1], 1, 1}},
            {"grid_dim", {input_shapes[0][0], 1, 1}},
            {"source_code", std::move(src)},
        });
    }
} // namespace

#include "nnfusion/core/kernels/cuda_gpu/inl/generate_kernel_code-inl.hpp"
