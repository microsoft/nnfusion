// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "batch_matmul.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::BatchMatMulMlas::BatchMatMulMlas(shared_ptr<KernelContext> ctx)
    : MlasKernelEmitter(ctx)
{
    arg0_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    arg1_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());

    std::stringstream tag;
    tag << "Mlas_batch_matmul"
        << "_i_" << join(arg0_shape, "_") << "_i_" << join(arg1_shape, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cpu::BatchMatMulMlas::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    auto generic_op = static_pointer_cast<nnfusion::op::GenericOp>(m_context->gnode->get_op_ptr());

    const nnfusion::Shape& input_shape_0 = m_context->inputs[0]->get_shape();
    const nnfusion::Shape& input_shape_1 = m_context->inputs[1]->get_shape();

    bool transA = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
    bool transB = generic_op->localOpConfig.getRoot()["adj_y"]["b"];
    size_t A1 = 1LU;
    for (int i = input_shape_0.size() - 3; i >= 0; --i)
        A1 *= input_shape_0[i];
    int A2, A3, A4, m, n, k, lda, stride_a, ldb, stride_b, ldc, stride_c;

    if (!transA && !transB)
    {
        A2 = input_shape_0[input_shape_0.size() - 2];
        A3 = input_shape_0[input_shape_0.size() - 1];
        A4 = input_shape_1[input_shape_1.size() - 1];
        m = A2, n = A4, k = A3, lda = A3, ldb = A4, ldc = A4;
    }
    else if (!transA && transB)
    {
        A2 = input_shape_0[input_shape_0.size() - 2];
        A3 = input_shape_0[input_shape_0.size() - 1];
        A4 = input_shape_1[input_shape_1.size() - 2];
        m = A2, n = A4, k = A3, lda = A3, ldb = input_shape_1[input_shape_1.size() - 1], ldc = A4;
    }
    else if (transA && !transB)
    {
        A2 = input_shape_0[input_shape_0.size() - 1];
        A3 = input_shape_0[input_shape_0.size() - 2];
        A4 = input_shape_1[input_shape_1.size() - 1];
        m = A2, n = A4, k = A3, lda = A2, ldb = A4, ldc = A4;
    }
    else
    { // transA && transB
        A2 = input_shape_0[input_shape_0.size() - 1];
        A3 = input_shape_0[input_shape_0.size() - 2];
        A4 = input_shape_1[input_shape_1.size() - 2];
        m = A2, n = A4, k = A3, lda = A2, ldb = input_shape_1[input_shape_1.size() - 1], ldc = A4;
    }
    auto code = op::create_code_from_template(
        R"(
int num_shards = static_cast<int64_t>(thread_pool->NumThreads());
const int32_t batch = @batch@;
const int64_t block_size = (batch + num_shards - 1) / num_shards;
if (block_size > batch)
{
    num_shards = 1;
}

auto func = [&](int __rank__){
    for (int b_inner = 0; b_inner < block_size; ++b_inner){
        if (((((int)__rank__) * block_size) + b_inner) < batch){
            MlasGemm(@transA@, @transB@, @m@, @n@, @k@, 1.0, input0+block_size*((int)__rank__)*@index0@+b_inner*@index0@, @lda@, input1+block_size*((int)__rank__)*@index1@+b_inner*@index1@, @ldb@, 0.0, output0+block_size*((int)__rank__)*@index2@+b_inner*@index2@, @ldc@, thread_pool);
        }
    }
};

thread_pool->ParallelFor(num_shards, func);
         
)",
        {{"m", m},
         {"n", n},
         {"k", k},
         {"lda", lda},
         {"ldb", ldb},
         {"ldc", ldc},
         {"transA", transA ? "CblasTrans" : "CblasNoTrans"},
         {"transB", transB ? "CblasTrans" : "CblasNoTrans"},
         {"index0",
          input_shape_0[input_shape_0.size() - 1] * input_shape_0[input_shape_0.size() - 2]},
         {"index1",
          input_shape_1[input_shape_1.size() - 1] * input_shape_1[input_shape_1.size() - 2]},
         {"index2", m * n},
         {"batch", A1}});

    lu << code;

    return _lu;
}

LanguageUnit_p cpu::BatchMatMulMlas::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::mlas);

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "BatchMatMul",                                                            // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mlas").Priority(6), // attrs
    cpu::BatchMatMulMlas)
