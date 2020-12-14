// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// #include "../cuda_cudnn.hpp"
#include "../cuda_cudnn.hpp"
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class DropoutTraining : public CudaLibEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                float ratio;

            public:
                DropoutTraining(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                    , ratio(generic_op->localOpConfig.getRoot()["ratio"])
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_signature() override
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
                       << "(cudnnHandle_t cudnn_handle, " << join(params, ", ") << ")";
                    return _lu;
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();

                    const nnfusion::Shape& input_shape = m_context->inputs[0]->get_shape();
                    auto shape_size = nnfusion::shape_size(input_shape);
                    float ratio = generic_op->localOpConfig.getRoot()["ratio"];

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    // move to global
                    // lu << "cudnnDropoutDescriptor_t dropout_desc;\n";
                    lu << "cudnnTensorDescriptor_t dropout_in_out_desc;\n";
                    // lu << "size_t dropout_state_size;\n";
                    // lu << "size_t dropout_reserve_size;\n";
                    // lu << "void* states;\n";
                    // lu << "CUDNN_SAFE_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));\n";
                    lu << "CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&dropout_in_out_desc));\n";
                    // lu << "CUDNN_SAFE_CALL(cudnnDropoutGetStatesSize(cudnn_handle, "
                    //       "&dropout_state_size));\n";

                    // lu << "CUDA_SAFE_CALL(cudaMalloc(&states, dropout_state_size));\n";
                    lu << "CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(dropout_in_out_desc, "
                          "CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, "
                       << static_cast<int>(shape_size) << ", 1, 1, 1));\n";
                    // The dropout_reserve_size is input_size/8, so mask will be bool[input_size]
                    // or char[input_size/8], not a real char[input_size],
                    // wwe cannot use it except in dropout backward, see detail in unit test
                    // lu << "CUDNN_SAFE_CALL(cudnnDropoutGetReserveSpaceSize(dropout_in_out_desc, "
                    //       "&dropout_reserve_size));\n";
                    // TODO: generate random seed
                    // lu << "CUDNN_SAFE_CALL(cudnnSetDropoutDescriptor(dropout_desc, cudnn_handle, "
                    //    << ratio << ", states, dropout_state_size, /*seed*/ 0));\n";
                    lu << "CUDNN_SAFE_CALL(cudnnDropoutForward(cudnn_handle,"
                       << " dropout_" << ratio2str(ratio) << "_desc,"
                       << " dropout_in_out_desc,"
                       << " input0,"
                       << " dropout_in_out_desc,"
                       << " output0,"
                       << " output1," << m_context->outputs[1]->size(true)
                       //    << " dropout_reserve_size"
                       << "));\n";

                    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(dropout_in_out_desc));\n";
                    // lu << "CUDNN_SAFE_CALL(cudnnDestroyDropoutDescriptor(dropout_desc));\n";
                    // lu << "CUDA_SAFE_CALL(cudaFree(states));\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::cudnn);
                    _lu->require(header::stdexcept);
                    _lu->require(header::sstream);
                    _lu->require(macro::CUDNN_SAFE_CALL);
                    _lu->require(cuda::get_dropout_global_states(ratio));
                    //_lu->require(declaration::cudnn_handle);
                    return _lu;
                }

                bool require_cudnn_handle() override { return true; }
            };

            class DropoutTrainingGrad : public CudaLibEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                float ratio;

            public:
                DropoutTrainingGrad(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                    , ratio(generic_op->localOpConfig.getRoot()["ratio"])
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_signature() override
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
                       << "(cudnnHandle_t cudnn_handle, " << join(params, ", ") << ")";
                    return _lu;
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();

                    const nnfusion::Shape& input_shape = m_context->inputs[0]->get_shape();
                    auto shape_size = nnfusion::shape_size(input_shape);

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    // lu << "cudnnDropoutDescriptor_t dropout_desc;\n";
                    lu << "cudnnTensorDescriptor_t dropout_in_out_desc;\n";
                    // lu << "size_t dropout_state_size;\n";
                    // lu << "size_t dropout_reserve_size;\n";
                    // lu << "void* states;\n";
                    // lu << "CUDNN_SAFE_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));\n";
                    lu << "CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&dropout_in_out_desc));\n";
                    // lu << "CUDNN_SAFE_CALL(cudnnDropoutGetStatesSize(cudnn_handle, "
                    //       "&dropout_state_size));\n";
                    // lu << "CUDA_SAFE_CALL(cudaMalloc(&states, dropout_state_size));\n";
                    lu << "CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(dropout_in_out_desc, "
                          "CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, "
                       << static_cast<int>(shape_size) << ", 1, 1, 1));\n";
                    // lu << "CUDNN_SAFE_CALL(cudnnDropoutGetReserveSpaceSize(dropout_in_out_desc, "
                    //       "&dropout_reserve_size));\n";
                    // TODO: generate random seed
                    // lu << "CUDNN_SAFE_CALL(cudnnSetDropoutDescriptor(dropout_desc, cudnn_handle, "
                    //    << ratio << ", states, dropout_state_size, /*seed*/ 0));\n";
                    lu << "CUDNN_SAFE_CALL(cudnnDropoutBackward(cudnn_handle,"
                       << " dropout_" << ratio2str(ratio) << "_desc,"
                       << " dropout_in_out_desc,"
                       << " input0,"
                       << " dropout_in_out_desc,"
                       << " output0,"
                       << " input1,"
                       //    << " dropout_reserve_size"
                       << m_context->inputs[1]->size(true) << "));\n";

                    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(dropout_in_out_desc));\n";
                    // lu << "CUDNN_SAFE_CALL(cudnnDestroyDropoutDescriptor(dropout_desc));\n";
                    // lu << "CUDA_SAFE_CALL(cudaFree(states));\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::cudnn);
                    _lu->require(header::stdexcept);
                    _lu->require(header::sstream);
                    _lu->require(macro::CUDNN_SAFE_CALL);
                    _lu->require(cuda::get_dropout_global_states(ratio));
                    //_lu->require(declaration::cudnn_handle);
                    return _lu;
                }

                bool require_cudnn_handle() override { return true; }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("DropoutTraining",                                          // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cudnn"), // attrs
                        cuda::DropoutTraining)                                      // constructor

REGISTER_KERNEL_EMITTER("DropoutTrainingGrad",                                      // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cudnn"), // attrs
                        cuda::DropoutTrainingGrad)                                  // constructor
