// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dot.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::Dot::Dot(shared_ptr<KernelContext> ctx)
    : EigenKernelEmitter(ctx)
{
    auto dot_op = static_pointer_cast<nnfusion::op::Dot>(ctx->gnode->get_op_ptr());

    reduction_axes = dot_op->get_reduction_axes_count();
    arg0_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    arg1_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());
    out_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());

    std::stringstream tag;
    tag << "_eigen_"
        << "_reduction_axes_count_" << reduction_axes << "_i_" << join(arg0_shape, "_") << "_i_"
        << join(arg1_shape, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cpu::Dot::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // void kernel(mcontext->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    lu << m_context->inputs[0]->get_element_type().c_type_string() << "* "
       << m_context->inputs[0]->get_name() << " = input0;\n";
    lu << m_context->inputs[1]->get_element_type().c_type_string() << "* "
       << m_context->inputs[1]->get_name() << " = input1;\n";
    lu << m_context->outputs[0]->get_element_type().c_type_string() << "* "
       << m_context->outputs[0]->get_name() << " = output0;\n";

    if (arg0_shape.empty() || arg1_shape.empty())
    {
        auto& first = (arg0_shape.empty() ? m_context->inputs[0] : m_context->inputs[1]);
        auto& second = (arg0_shape.empty() ? m_context->inputs[1] : m_context->inputs[0]);

        lu.block_begin();
        lu << emit_eigen_vector(m_context->outputs[0]) << "\n    = ";
        lu << first->get_name() << "[0]\n    * " << emit_eigen_vector(second) << ";\n";
        lu.block_end();
    }
    else if ((arg0_shape.size() == 1) && (arg1_shape.size() == 1) && reduction_axes == 1)
    {
        lu.block_begin();
        lu << emit_eigen_vector(m_context->outputs[0]) << " = \n"
           << "    " << emit_eigen_vector(m_context->inputs[0]) << ".dot("
           << emit_eigen_vector(m_context->inputs[1]) << ");\n";
        lu.block_end();
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1) && reduction_axes == 1)
    {
        lu.block_begin();
        lu << emit_eigen_vector(m_context->outputs[0]) << " = \n"
           << "    " << emit_eigen_matrix(m_context->inputs[0]) << " * "
           << emit_eigen_vector(m_context->inputs[1]) << ";\n";
        lu.block_end();
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2) && reduction_axes == 1)
    {
        lu.block_begin();
        lu << emit_eigen_matrix(m_context->outputs[0]) << " = \n"
           << "    " << emit_eigen_matrix(m_context->inputs[0]) << " * "
           << emit_eigen_matrix(m_context->inputs[1]) << ";\n";
        lu.block_end();
    }
    else
    {
        return nullptr;
    }

    return _lu;
}

LanguageUnit_p cpu::Dot::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::eigen_utils);

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Dot",                                                                     // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("eigen").Priority(4), // attrs
    cpu::Dot)
