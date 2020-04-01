// Microsoft (c) 2019, NNFusion Team

// This is the 2rd-generation of kernel definition, recommend to extend new ops with this style
// Changes needed for creating an new kernel with 2rd generation style:
//
// files to change:
//   [a] ./new_kernel_0.cpp
//   [b] ../../../ops/op_define/new_op_0.cpp

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

/*********************************
>> For Config detail, please reference ../../../ops/op_define/BatchMatMul.cpp

Example:
    BatchMatMul::Config {
        "adj_x": {
            "b": false,
        },
        "adj_y": {
            "b": false,
        },
    }
*********************************/

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class BatchMatMul : public CudaLibEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;

            public:
                BatchMatMul(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();

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
                        m = A4, n = A2, k = A3, lda = A4, stride_a = A3 * A4, ldb = A3,
                        stride_b = A2 * A3, ldc = A4, stride_c = A2 * A4;
                    }
                    else if (!transA && transB)
                    {
                        A2 = input_shape_0[input_shape_0.size() - 2];
                        A3 = input_shape_0[input_shape_0.size() - 1];
                        A4 = input_shape_1[input_shape_1.size() - 2];
                        m = A4, n = A2, k = A3, lda = A3, stride_a = A3 * A4, ldb = A3,
                        stride_b = A2 * A3, ldc = A4, stride_c = A2 * A4;
                    }
                    else if (transA && !transB)
                    {
                        A2 = input_shape_0[input_shape_0.size() - 1];
                        A3 = input_shape_0[input_shape_0.size() - 2];
                        A4 = input_shape_1[input_shape_1.size() - 1];
                        m = A4, n = A2, k = A3, lda = A4, stride_a = A3 * A4, ldb = A2,
                        stride_b = A2 * A3, ldc = A4, stride_c = A2 * A4;
                    }
                    else
                    { // transA && transB
                        A2 = input_shape_0[input_shape_0.size() - 1];
                        A3 = input_shape_0[input_shape_0.size() - 2];
                        A4 = input_shape_1[input_shape_1.size() - 2];
                        m = A4, n = A2, k = A3, lda = A3, stride_a = A3 * A4, ldb = A2,
                        stride_b = A2 * A3, ldc = A4, stride_c = A2 * A4;
                    }

                    float alpha = 1.0f, beta = 0.0f;
                    auto code = nnfusion::op::create_code_from_template(
                        R"(
                        static const float alpha = @alpha@F, beta = @beta@F;
                        if (!@hCublas@)
                            CUBLAS_SAFE_CALL(@api_create@(&@hCublas@));
                        CUBLAS_SAFE_CALL(cublasSetStream(@hCublas@, stream));
                        CUBLAS_SAFE_CALL(@api_exec@(
                            @hCublas@, @transA@, @transB@, @m@, @n@, @k@,
                            &alpha, input1, @lda@, @stride_a@, input0, @ldb@, @stride_b@,
                            &beta, output0, @ldc@, @stride_c@, @batch@));
                    )",
                        {
                            {"hCublas", "global_cublas_handle"},
                            {"api_create", "cublasCreate"},
                            {"api_exec", "cublasSgemmStridedBatched"},
                            {"transA", transB ? "CUBLAS_OP_T" : "CUBLAS_OP_N"},
                            {"transB", transA ? "CUBLAS_OP_T" : "CUBLAS_OP_N"},
                            {"alpha", alpha},
                            {"beta", beta},
                            {"m", m},
                            {"n", n},
                            {"k", k},
                            {"lda", lda},
                            {"ldb", ldb},
                            {"ldc", ldc},
                            {"stride_a", stride_a},
                            {"stride_b", stride_b},
                            {"stride_c", stride_c},
                            {"batch", A1},
                        });

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    // function signature:
                    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
                    lu.block_begin();
                    lu << code << "\n";
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu_header(new LanguageUnit(get_function_name() + "_dep"));
                    _lu_header->require(header::cuda);
                    _lu_header->require(header::cublas);
                    _lu_header->require(declaration::global_cublas_handle);
                    _lu_header->require(macro::CUBLAS_SAFE_CALL);
                    return _lu_header;
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register Pad kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("BatchMatMul",                                                // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::BatchMatMul)                                            // constructor

REGISTER_KERNEL_EMITTER("BatchMatMul",                                                // op_name
                        Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::BatchMatMul)                                            // constructor
