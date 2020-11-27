// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dot.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::DotMkl::DotMkl(shared_ptr<KernelContext> ctx)
    : MklKernelEmitter(ctx)
{
    auto dot_op = static_pointer_cast<nnfusion::op::Dot>(ctx->gnode->get_op_ptr());

    reduction_axes = dot_op->get_reduction_axes_count();
    arg0_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    arg1_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());
    out_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());

    std::stringstream tag;
    tag << "mklblas"
        << "_r_" << reduction_axes << "_i_" << join(arg0_shape, "_") << "_i_"
        << join(arg1_shape, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cpu::DotMkl::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // void kernel(mcontext->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2) && reduction_axes == 1)
    {
        lu << "cblas_sgemm("
           << "CblasRowMajor, "
           << "CblasNoTrans, "
           << "CblasNoTrans, " << arg0_shape[0] << ", " << arg1_shape[1] << ", " << arg0_shape[1]
           << ",\n"
           << "        1.0f, "
           << "input0, " << max(1UL, arg0_shape[1]) << ", "
           << "input1, " << max(1UL, arg1_shape[1]) << ", 0.0f, "
           << "output0, " << max(1UL, arg1_shape[1]) << ");\n";
    }
    else if ((arg0_shape.size() == 3) && (arg1_shape.size() == 2) && reduction_axes == 1)
    {
        auto& mat_a = m_context->inputs[0];
        auto& mat_b = m_context->inputs[1];
        auto& mat_c = m_context->outputs[0];
        const Shape& shape_a = mat_a->get_shape();
        const Shape& shape_b = mat_b->get_shape();

        const size_t m = shape_a[1];
        const size_t k = shape_a[2];
        const size_t n = shape_b[1];

        // this also works when mat_a is shape (1, m, k)
        const size_t offset_a = m * k;
        // we do not offset mat_b
        const size_t offset_b = 0;
        const size_t offset_c = m * n;

        const size_t group_count = 1;
        const size_t group_size = shape_a[0];
        auto populate_array = [&lu](const std::string& var, size_t size, size_t offset) {
            for (size_t i = 0; i < size; ++i)
            {
                lu << var << "+" << i * offset << ((i < size - 1) ? ", " : "");
            }
        };

        lu << "CBLAS_TRANSPOSE transa_array[] = {CblasNoTrans};\n";
        lu << "CBLAS_TRANSPOSE transb_array[] = {CblasNoTrans};\n";
        lu << "int m_array[] = {" << m << "};\n";
        lu << "int n_array[] = {" << n << "};\n";
        lu << "int k_array[] = {" << k << "};\n";
        lu << "float alpha_array[] = {1.0f};\n";
        lu << "std::vector<const float*> a{";
        populate_array("input0", group_size, offset_a);
        lu << "};\n";
        lu << "const float** a_array = &a[0];\n";
        lu << "int lda_array[] = {" << std::max(1UL, k) << "};\n";
        lu << "std::vector<const float*> b{";
        populate_array("input1", group_size, offset_b);
        lu << "};\n";
        lu << "const float** b_array = &b[0];\n";
        lu << "int ldb_array[] = {" << std::max(1UL, n) << "};\n";
        lu << "float beta_array[] = {0.0f};\n";
        lu << "std::vector<float*> c{";
        populate_array("output0", group_size, offset_c);
        lu << "};\n";
        lu << "float** c_array = &c[0];\n";
        lu << "int ldc_array[] = {" << std::max(1UL, n) << "};\n";
        lu << "int group_size[] = {" << group_size << "};\n";

        lu << "cblas_sgemm_batch(CblasRowMajor, ";
        lu << "transa_array, transb_array, m_array, n_array, k_array, \n";
        lu << "alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, \n";
        lu << "c_array, ldc_array, " << group_count << ", group_size);\n";
    }
    else if ((arg0_shape.size() == 4) && (arg0_shape[0] == 1) && (arg1_shape.size() == 2) &&
             reduction_axes == 1)
    {
        auto& mat_a = m_context->inputs[0];
        auto& mat_b = m_context->inputs[1];
        auto& mat_c = m_context->outputs[0];
        const Shape& shape_a = mat_a->get_shape();
        const Shape& shape_b = mat_b->get_shape();

        const size_t m = shape_a[2];
        const size_t k = shape_a[3];
        const size_t n = shape_b[1];

        const size_t offset_a = m * k;
        // we do not offset mat_b
        const size_t offset_b = 0;
        const size_t offset_c = m * n;

        const size_t group_count = 1;
        const size_t group_size = shape_a[1];
        auto populate_array = [&lu](const std::string& var, size_t size, size_t offset) {
            for (size_t i = 0; i < size; ++i)
            {
                lu << var << "+" << i * offset << ((i < size - 1) ? ", " : "");
            }
        };

        lu << "CBLAS_TRANSPOSE transa_array[] = {CblasNoTrans};\n";
        lu << "CBLAS_TRANSPOSE transb_array[] = {CblasNoTrans};\n";
        lu << "int m_array[] = {" << m << "};\n";
        lu << "int n_array[] = {" << n << "};\n";
        lu << "int k_array[] = {" << k << "};\n";
        lu << "float alpha_array[] = {1.0f};\n";
        lu << "std::vector<const float*> a{";
        populate_array("input0", group_size, offset_a);
        lu << "};\n";
        lu << "const float** a_array = &a[0];\n";
        lu << "int lda_array[] = {" << std::max(1UL, k) << "};\n";
        lu << "std::vector<const float*> b{";
        populate_array("input1", group_size, offset_b);
        lu << "};\n";
        lu << "const float** b_array = &b[0];\n";
        lu << "int ldb_array[] = {" << std::max(1UL, n) << "};\n";
        lu << "float beta_array[] = {0.0f};\n";
        lu << "std::vector<float*> c{";
        populate_array("output0", group_size, offset_c);
        lu << "};\n";
        lu << "float** c_array = &c[0];\n";
        lu << "int ldc_array[] = {" << std::max(1UL, n) << "};\n";
        lu << "int group_size[] = {" << group_size << "};\n";

        lu << "cblas_sgemm_batch(CblasRowMajor, ";
        lu << "transa_array, transb_array, m_array, n_array, k_array, \n";
        lu << "alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, \n";
        lu << "c_array, ldc_array, " << group_count << ", group_size);\n";
    }
    else
    {
        return nullptr;
    }

    return _lu;
}

LanguageUnit_p cpu::DotMkl::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cblas);

    if ((arg0_shape.size() == 3) && (arg1_shape.size() == 2) && reduction_axes == 1)
    {
        _lu->require(header::vector);
    }

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Dot",                                                                   // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mkl").Priority(3), // attrs
    cpu::DotMkl)
