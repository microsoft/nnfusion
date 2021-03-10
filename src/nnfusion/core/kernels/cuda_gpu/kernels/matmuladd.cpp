// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "matmuladd.hpp"
#include "../cuda_common_ops.hpp"
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::MatMulAdd::MatMulAdd(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    generic_op = static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr());
    A_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    B_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());
    C_shape = nnfusion::Shape(ctx->inputs[2]->get_shape());
    out_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());
    tmp_tensor = allocate_tensor(m_context->inputs[2]->get_shape(), dtype);
    std::stringstream tag;
    tag << "cublas_MatMulAdd"
        << "_dtype_" << dtype.c_type_string() << "_i_" << join(A_shape, "_") << "_i_"
        << join(B_shape, "_") << "_i_" << join(C_shape, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::MatMulAdd::emit_function_body()
{
    auto& ctx = m_context;
    auto& cfg = generic_op->localOpConfig.getRoot();
    bool trans_A = cfg["trans_A"];
    bool trans_B = cfg["trans_B"];
    size_t m = trans_A ? A_shape[1] : A_shape[0];
    size_t k = trans_A ? A_shape[0] : A_shape[1];
    size_t n = trans_B ? B_shape[0] : B_shape[1];
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    auto add_kernel = get_add_kernel();
    lu.require(add_kernel);
    auto code = nnfusion::op::create_code_from_template(
        R"(
if (output0 != input2)
{
    cudaStream_t stream = nullptr;
    CUBLAS_SAFE_CALL(cublasGetStream(cublas_handle, &stream));
    const @dtype@ alpha = 1.0;
    const @dtype@ beta = 0;
    CUBLAS_SAFE_CALL(@cublasgemm@(cublas_handle, 
                    @transb@, 
                    @transa@,
                    @n@, @m@, @k@, &alpha, 
                    static_cast<const @dtype@*>(input1), @ldb@,
                    static_cast<const @dtype@*>(input0), @lda@,
                    &beta, static_cast<@dtype@*>(@tmp_tensor@), @n@));
    @add_function_call@<<<dim3(@gx@, @gy@, @gz@), dim3(@bx@, @by@, @bz@), 0, stream>>>(@tmp_tensor@, input2, output0);  
}
else
{
    const @dtype@ alpha = 1.0;
    const @dtype@ beta = 1.0;
    CUBLAS_SAFE_CALL(@cublasgemm@(cublas_handle, 
                    @transb@, 
                    @transa@,
                    @n@, @m@, @k@, &alpha, 
                    static_cast<const @dtype@*>(input1), @ldb@,
                    static_cast<const @dtype@*>(input0), @lda@,
                    &beta, static_cast<@dtype@*>(input2), @n@));
}
    )",
        {{"cublasgemm", (dtype == element::f16) ? "cublasHgemm" : "cublasSgemm"},
         {"transb", trans_B ? "CUBLAS_OP_T" : "CUBLAS_OP_N"},
         {"transa", trans_A ? "CUBLAS_OP_T" : "CUBLAS_OP_N"},
         {"dtype", (dtype == element::f16) ? "half" : "float"},
         {"n", n},
         {"m", m},
         {"k", k},
         {"ldb", trans_B ? k : n},
         {"lda", trans_A ? m : k},
         {"buffer", m_context->outputs[0]->size()},
         {"tmp_tensor", tmp_tensor->get_name()},
         {"add_function_call", add_kernel->get_symbol()},
         {"gx", m_gridDim.x},
         {"gy", m_gridDim.y},
         {"gz", m_gridDim.z},
         {"bx", m_blockDim.x},
         {"by", m_blockDim.y},
         {"bz", m_blockDim.z}});

    lu << code << "\n";
    return _lu;
}

LanguageUnit_p cuda::MatMulAdd::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cublas);
    _lu->require(header::stdexcept);
    _lu->require(header::sstream);
    _lu->require(macro::CUBLAS_SAFE_CALL);
    _lu->require(macro::CUDA_SAFE_CALL);
    // _lu->require(declaration::cuda_fp16_scale);
    //_lu->require(declaration::cublas_handle);
    return _lu;
}

LanguageUnit_p cuda::MatMulAdd::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
    auto& lu = *_lu;

    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
        ss << "input" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->tensors.size(); i++)
    {
        stringstream ss;
        ss << m_context->tensors[i]->get_element_type().c_type_string() << "* ";
        // defult name is: "persit0", "persist1" ...
        ss << m_context->tensors[i]->get_name();
        params.push_back(ss.str());
    }

    lu << "void "
       << "(cublasHandle_t cublas_handle, " << join(params, ", ") << ")";
    return _lu;
}

LanguageUnit_p cuda::MatMulAdd::get_add_kernel()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_add"));
    auto& lu = *_lu;
    std::vector<string> data_types{
        dtype.c_type_string(), dtype.c_type_string(), dtype.c_type_string()};
    auto iter = CudaElementOpMap.find("Add");
    auto math_kernel = get_math_kernel(iter->second.op, iter->second.math_kernel, data_types);
    NNFUSION_CHECK_NOT_NULLPTR(math_kernel);
    _lu->require(math_kernel);
    auto num_inputs = data_types.size() - 1;
    uint32_t nthreads =
        static_cast<uint32_t>(nnfusion::shape_size(m_context->outputs[0]->get_shape()));
    int grids, blocks, bound;
    uint32_t num_ele =
        static_cast<uint32_t>(nnfusion::shape_size(m_context->outputs[0]->get_shape()));

    auto compute_best_config = [&](int& grids, int& blocks, int& bound) {
        uint32_t num_ele =
            static_cast<uint32_t>(nnfusion::shape_size(m_context->outputs[0]->get_shape()));
        for (int i = 512; i >= 64; i >>= 1)
        {
            if (num_ele % i == 0)
            {
                grids = num_ele / i, blocks = i, bound = 0;
                return;
            }
        }
        for (int i = 512; i >= 32; i--)
        {
            if (num_ele % i == 0)
            {
                grids = num_ele / i, blocks = i, bound = 0;
                return;
            }
        }
        if (num_ele < 32)
            grids = 1, blocks = num_ele, bound = 0;
        else
            grids = (num_ele + 255) / 256, blocks = 256, bound = 1;
    };

    compute_best_config(grids, blocks, bound);
    m_blockDim = dim3(blocks, 1, 1);
    m_gridDim = dim3(grids, 1, 1);
    string params = dtype.c_type_string() + "* input0, " + dtype.c_type_string() + "* input1, " +
                    dtype.c_type_string() + "* output0";
    std::string name = "MatMulAdd_add_" + to_string(m_blockDim.x) + "_" + to_string(m_gridDim.x);
    lu << "extern \"C\" __launch_bounds__(" << m_blockDim.x * m_blockDim.y * m_blockDim.z
       << ") __global__ void " << name << "(" << params << ")";
    lu.block_begin();
    {
        std::string tid = "blockIdx.x * " + std::to_string(blocks) + " + threadIdx.x";
        if (grids == 1)
            tid = "threadIdx.x";
        if (bound)
            lu << "if (" << tid << " >= " << bound << ") return;";
        {
            std::string invoke_func = iter->second.op;
            ;
            lu << "output0[" << tid << "] = " << invoke_func << "(";
            for (size_t i = 0; i < num_inputs - 1; i++)
            {
                lu << "input" << i << "[" << tid << "], ";
            }
            lu << "input" << num_inputs - 1 << "[" << tid << "]);\n";
        }
    }

    lu.block_end();
    lu.change_symbol(name);
    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "MatMulAdd",                                                             // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cublas").Priority(2), // attrs
    cuda::MatMulAdd)                                                         // constructor

REGISTER_KERNEL_EMITTER(
    "MatMulAdd",                                                             // op_name
    Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cublas").Priority(2), // attrs
    cuda::MatMulAdd)                                                         // constructor

REGISTER_KERNEL_EMITTER(
    "MatMulAdd",                                                             // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cublas").Priority(2), // attrs
    cuda::MatMulAdd)                                                         // constructor
