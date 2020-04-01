// Microsoft (c) 2019, NNFusion Team

#include "dot.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Dot::Dot(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto dot_op = static_pointer_cast<nnfusion::op::Dot>(ctx->gnode->get_op_ptr());

    reduction_axes = dot_op->get_reduction_axes_count();
    arg0_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    arg1_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());
    out_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());

    std::stringstream tag;
    tag << "cublas_dot"
        << "_dtype_" << dtype.c_type_string() << "_reduction_axes_count_" << reduction_axes << "_i_"
        << join(arg0_shape, "_") << "_i_" << join(arg1_shape, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Dot::emit_function_body()
{
    auto& ctx = m_context;
    auto gemm = static_pointer_cast<nnfusion::op::Dot>(ctx->gnode->get_op_ptr());
    auto trans_A = gemm->get_transpose_A();
    auto trans_B = gemm->get_transpose_B();

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)

    //lu.block_begin();
    {
        lu << "CUBLAS_SAFE_CALL(cublasSetStream(global_cublas_handle, stream));\n";
        // case 1: Scalar * Tensor
        if (arg0_shape.empty() || arg1_shape.empty())
        {
            auto& second = (arg0_shape.empty() ? arg1_shape : arg0_shape);
            size_t count = nnfusion::shape_size(second);

            string firstarg = (arg0_shape.empty() ? "input0" : "input1");
            string secondarg = (arg0_shape.empty() ? "input1" : "input0");

            lu << "CUBLAS_SAFE_CALL(cublasScopy(global_cublas_handle, " << count
               << ", static_cast<const float*>(" << firstarg
               << "), 1, static_cast<float*>(output0),1));\n";
            lu << "CUBLAS_SAFE_CALL(cublasSscal(global_cublas_handle, " << count
               << ", static_cast<const float*>(" << secondarg
               << ", static_cast<float*>(output0),1));\n";
        }
        // case 2: 1d Dot
        else if ((arg0_shape.size() == arg1_shape.size()) && (arg0_shape.size() == reduction_axes))
        {
            for (int i = 0; i < arg0_shape.size(); i++)
            {
                if (arg0_shape[i] != arg1_shape[i])
                {
                    std::vector<std::string> arg_vec{"arg0", "arg1"};
                    std::vector<nnfusion::Shape> shape_vec{arg0_shape, arg1_shape};

                    CHECK_FAIL() << nnfusion::join(arg_vec) << " with " << nnfusion::join(shape_vec)
                                 << " respectively, at Node " << m_context->gnode->get_name()
                                 << ", do not match for dot op";
                }
            }

            size_t count = nnfusion::shape_size(arg0_shape);
            lu << "CUBLAS_SAFE_CALL(cublasSdot(global_cublas_handle, " << count
               << ", static_cast<const float*>(input0), 1, static_cast<const float*>(input0), 1, "
                  "static_cast<float*>(output0)));\n";
        }
        // matrix * vector
        else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1) && (reduction_axes == 1))
        {
            /*
            lu << "const float alpha = 1.0;\n const float beta = 0;\n";
              <<"CUBLAS_SAFE_CALL(
                    cublasSetPointerMode(global_cublas_handle, CUBLAS_POINTER_MODE_HOST));
              \n ";
              */
            lu << "CUBLAS_SAFE_CALL(cublasSgemv(global_cublas_handle, ";
            if (trans_A)
                lu << "CUBLAS_OP_N, " << arg0_shape[0] << ", " << arg0_shape[1] << ", ";
            else
                lu << "CUBLAS_OP_T, " << arg0_shape[1] << ", " << arg0_shape[0] << ", ";
            lu << " &alpha,"
               << " static_cast<const float*>(input0)," << arg0_shape[1] << ", "
               << " static_cast<const float*>(input1),"
               << " 1,"
               << " &beta,"
               << " static_cast<float*>(output0),"
               << " 1));\n";
        }
        else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2) && (reduction_axes == 1) &&
                 (trans_A || trans_B))
        {
            int m = trans_B ? arg1_shape[0] : arg1_shape[1];
            int n = trans_A ? arg0_shape[1] : arg0_shape[0];
            int k = trans_A ? arg0_shape[0] : arg0_shape[1];

            lu << "const float alpha = 1.0;\nconst float beta = 0;\n";

            lu << "CUBLAS_SAFE_CALL(cublasSgemm(global_cublas_handle,"
               << (trans_B ? " CUBLAS_OP_T," : " CUBLAS_OP_N,")
               << (trans_A ? " CUBLAS_OP_T," : " CUBLAS_OP_N,") << " " << m << ","
               << " " << n << ","
               << " " << k << ","
               << " &alpha,"
               << " static_cast<const float*>(input1),"
               << " " << arg1_shape[1] << ","
               << " static_cast<const float*>(input0),"
               << " " << arg0_shape[1] << ","
               << " &beta,"
               << " static_cast<float*>(output0),"
               << " " << m << "));\n";
        }
        else
        {
            size_t axes_for_m_count = arg0_shape.size() - reduction_axes;
            size_t axes_for_n_count = arg1_shape.size() - reduction_axes;
            size_t axes_for_k_count = reduction_axes;
            size_t m = 1;
            size_t n = 1;
            size_t k = 1;

            // check if input and output size correct
            // check and calculate k for arg0 and arg1
            size_t arg0_k_idx = axes_for_m_count; // first axe in arg0 for k
            size_t arg1_k_idx = 0;                // first axe in arg1 for k

            for (size_t i = 0; i < axes_for_k_count; i++)
            {
                k *= arg0_shape[arg0_k_idx];
                if (arg0_shape[arg0_k_idx++] != arg1_shape[arg1_k_idx++])
                {
                    std::vector<std::string> arg_vec{"arg0", "arg1"};
                    std::vector<nnfusion::Shape> shape_vec{arg0_shape, arg1_shape};

                    CHECK_FAIL() << nnfusion::join(arg_vec) << " with " << nnfusion::join(shape_vec)
                                 << " respectively, at Node " << m_context->gnode->get_name()
                                 << ", do not match for dot op";
                }
            }
            // check and calculate m for arg0 and out
            size_t arg0_m_idx = 0; // first axe in arg0 for m
            size_t out_m_idx = 0;  // first axe in out for m
            for (size_t i = 0; i < axes_for_m_count; i++)
            {
                m *= arg0_shape[arg0_m_idx];
                if (arg0_shape[arg0_m_idx++] != out_shape[out_m_idx++])
                {
                    std::vector<std::string> arg_vec{"arg0", "output"};
                    std::vector<nnfusion::Shape> shape_vec{arg0_shape, out_shape};

                    CHECK_FAIL() << nnfusion::join(arg_vec) << " with " << nnfusion::join(shape_vec)
                                 << " respectively, at Node " << m_context->gnode->get_name()
                                 << ", do not match for dot op";
                }
            }
            // check and calculate n for arg1 and out
            size_t arg1_n_idx = axes_for_k_count; // first axe in arg1 for n
            size_t out_n_idx = axes_for_m_count;  // first axe in arg1 for n
            for (size_t i = 0; i < axes_for_n_count; i++)
            {
                n *= arg1_shape[arg1_n_idx];
                if (arg1_shape[arg1_n_idx++] != out_shape[out_n_idx++])
                {
                    std::vector<std::string> arg_vec{"arg1", "output"};
                    std::vector<nnfusion::Shape> shape_vec{arg1_shape, out_shape};

                    CHECK_FAIL() << nnfusion::join(arg_vec) << " with " << nnfusion::join(shape_vec)
                                 << " respectively, at Node " << m_context->gnode->get_name()
                                 << ", do not match for dot op";
                }
            }

            lu << "const float alpha = 1.0;\nconst float beta = 0;\n";

            lu << "CUBLAS_SAFE_CALL(cublasSgemm(global_cublas_handle,"
               << " CUBLAS_OP_N,"
               << " CUBLAS_OP_N,"
               << " " << n << ","
               << " " << m << ","
               << " " << k << ","
               << " &alpha,"
               << " static_cast<const float*>(input1),"
               << " " << n << ","
               << " static_cast<const float*>(input0),"
               << " " << k << ","
               << " &beta,"
               << " static_cast<float*>(output0),"
               << " " << n << "));\n";
        }
    }
    //lu.block_end();
    return _lu;
}

LanguageUnit_p cuda::Dot::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cublas);
    _lu->require(header::stdexcept);
    _lu->require(header::sstream);
    _lu->require(macro::CUBLAS_SAFE_CALL);
    _lu->require(declaration::global_cublas_handle);
    return _lu;
}

REGISTER_KERNEL_EMITTER("Dot",                                                   // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cublas"), // attrs
                        cuda::Dot)                                               // constructor

REGISTER_KERNEL_EMITTER("Dot",                                                   // op_name
                        Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("cublas"), // attrs
                        cuda::Dot)                                               // constructor
