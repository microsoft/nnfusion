// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dot.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::DotMlas::DotMlas(shared_ptr<KernelContext> ctx)
    : MlasKernelEmitter(ctx)
{
    auto dot_op = static_pointer_cast<op::Dot>(ctx->gnode->get_op_ptr());

    reduction_axes = dot_op->get_reduction_axes_count();
    arg0_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    arg1_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());

    std::stringstream tag;
    tag << "Mlas"
        << "_r_" << reduction_axes << "_i_" << join(arg0_shape, "_") << "_i_"
        << join(arg1_shape, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cpu::DotMlas::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    auto gemm = static_pointer_cast<nnfusion::op::Dot>(m_context->gnode->get_op_ptr());
    auto trans_A = gemm->get_transpose_A();
    auto trans_B = gemm->get_transpose_B();

    std::string trans_A_str = (trans_A) ? "CblasTrans" : "CblasNoTrans";
    std::string trans_B_str = (trans_B) ? "CblasTrans" : "CblasNoTrans";

    size_t M, K, N;

    if (arg0_shape.empty() || arg1_shape.empty())
    {
        M = (arg0_shape.empty()) ? 1 : nnfusion::shape_size(arg0_shape);
        N = (arg1_shape.empty()) ? 1 : nnfusion::shape_size(arg1_shape);
        K = 1;
    }
    else if ((arg0_shape.size() == arg1_shape.size()) && (arg0_shape.size() == reduction_axes))
    {
        for (int i = 0; i < arg0_shape.size(); i++)
        {
            if (arg0_shape[i] != arg1_shape[i])
            {
                std::vector<std::string> arg_vec{"arg0", "arg1"};
                std::vector<nnfusion::Shape> shape_vec{arg0_shape, arg1_shape};

                NNFUSION_CHECK_FAIL() << nnfusion::join(arg_vec) << " with "
                                      << nnfusion::join(shape_vec) << " respectively, at Node "
                                      << m_context->gnode->get_name()
                                      << ", do not match for dot op";
            }
        }

        M = 1;
        N = 1;
        K = nnfusion::shape_size(arg0_shape);
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2) && reduction_axes == 1)
    {
        M = (trans_A) ? arg0_shape[1] : arg0_shape[0];
        K = (trans_A) ? arg0_shape[0] : arg0_shape[1];
        N = (trans_B) ? arg1_shape[0] : arg1_shape[1];
        size_t arg1_K = (trans_B) ? arg1_shape[1] : arg1_shape[0];

        if (K != arg1_K)
        {
            std::vector<std::string> arg_vec{"arg0", "arg1"};
            std::vector<nnfusion::Shape> shape_vec{arg0_shape, arg1_shape};

            NNFUSION_CHECK_FAIL() << nnfusion::join(arg_vec) << " with "
                                  << nnfusion::join(shape_vec) << " respectively, at Node "
                                  << m_context->gnode->get_name() << ", do not match for dot op."
                                  << "transpose_A: " << trans_A_str
                                  << ", transpose_B: " << trans_B_str;
        }
    }
    else
    {
        return nullptr;
    }

    size_t lda = (trans_A) ? M : K;
    size_t ldb = (trans_B) ? K : N;
    size_t ldc = N;

    lu << "MlasGemm(" << ((trans_A) ? "CblasTrans, " : "CblasNoTrans, ")
       << ((trans_B) ? "CblasTrans, " : "CblasNoTrans, ") << M << ", " << N << ", " << K << ", "
       << "1.0, "
       << "input0, " << lda << ", "
       << "input1, " << ldb << ", "
       << "0.0, "
       << "output0, " << ldc << ", "
       << "thread_pool);";

    return _lu;
}

LanguageUnit_p cpu::DotMlas::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::mlas);

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Dot",                                                                    // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mlas").Priority(6), // attrs
    cpu::DotMlas)
