// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "avg_pool.hpp"
#include "../cuda_cudnn.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::pooling_op_shape cuda::AvgPool1D::avgpool_shape(nnfusion::Shape in,
                                                      nnfusion::Shape out,
                                                      nnfusion::Shape window,
                                                      nnfusion::Shape strides,
                                                      nnfusion::Shape pad)
{
    cuda::pooling_op_shape shape;
    shape.N = in[0];
    shape.C = in[1];
    shape.K = shape.C; // pooling feature maps is
    shape.J = shape.C; // not currently supported
    NNFUSION_CHECK(in.size() == 3) << "AvgPool1D require 1 spatial dimension.";

    shape.D = 1;
    shape.H = 1;
    shape.W = in[2];
    shape.M = 1;
    shape.P = 1;
    shape.Q = out[2];
    shape.T = 1;
    shape.R = 1;
    shape.S = window[0];
    shape.STRIDE_D = 0;
    shape.STRIDE_H = 0;
    shape.STRIDE_W = strides[0];
    shape.PAD_D = 0;
    shape.PAD_H = 0;
    shape.PAD_W = pad[0];

    return shape;
}

cuda::AvgPool1D::AvgPool1D(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    auto avg_pool = static_pointer_cast<nnfusion::op::AvgPool>(ctx->gnode->get_op_ptr());
    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    window_shape = nnfusion::Shape(avg_pool->get_window_shape());
    padding_below = nnfusion::Shape(avg_pool->get_padding_below());
    padding_above = nnfusion::Shape(avg_pool->get_padding_above());
    window_stride = nnfusion::Strides(avg_pool->get_window_movement_strides());
    include_pad = avg_pool->get_include_padding_in_avg_computation();
    input_type = ctx->inputs[0]->get_element_type();
    output_type = ctx->outputs[0]->get_element_type();

    // NNFUSION_CHECK(input_shape.size() == 3)
    //     << "Input shape size of AvgPool1D is invalid, shape size: " << input_shape.size()
    //     << "expected 3";

    std::stringstream tag;
    tag << "cuda_avgpool"
        << "_s" << join(input_shape, "_") << "_r" << join(output_shape, "_") << "_st"
        << join(window_stride, "_") << "_ip" << int(include_pad);
    custom_tag = tag.str();
}

LanguageUnit_p cuda::AvgPool1D::emit_function_body()
{
    if (input_shape.size() != 3)
        return nullptr;

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    shape = cuda::AvgPool1D::avgpool_shape(
        input_shape, output_shape, window_shape, window_stride, padding_below);

    // Precompute for fast constant memory access.
    HW = shape.H * shape.W;
    DHW = shape.D * HW;
    CDHW = shape.C * DHW;
    PQ = shape.P * shape.Q;
    MPQ = shape.M * PQ;
    KMPQ = shape.K * MPQ;
    RS = shape.R * shape.S;
    TRS = shape.T * RS;

    // Precompute magic numbers and shifts for fast integer division.
    std::tie(magic_N, shift_N) = idiv_magic_u64(shape.N);
    std::tie(magic_P, shift_P) = idiv_magic_u64(shape.P);
    std::tie(magic_S, shift_S) = idiv_magic_u64(shape.S);
    std::tie(magic_RS, shift_RS) = idiv_magic_u64(RS);

    // TODO: blending factors are not currently implemented
    alpha = 1.0f;
    beta = 0.0f;

    /*
    << "float alpha, float beta, "
    << "int N, int C, int D, int H, int W, "
    << "int HW, int DHW, int CDHW, int magic_N, int shift_N, "
    << "int P, int Q, int magic_P, int shift_P, "
    << "int PQ, int MPQ, int KMPQ, "
    << "int S, int RS, int TRS, "
    << "int magic_S, int shift_S, int magic_RS, int shift_RS, "
    << "int str_d, int str_h, int str_w, "
    << "int pad_d, int pad_h, int pad_w"
    << ")\n";
    */
    /*CONST*/
    lu << "float alpha = " << alpha << ";\n";
    lu << "float beta = " << beta << ";\n";
    lu << "int N = " << shape.N << ";\n";
    lu << "int C = " << shape.C << ";\n";
    lu << "int D = " << shape.D << ";\n";
    lu << "int H = " << shape.H << ";\n";
    lu << "int W = " << shape.W << ";\n";

    lu << "int HW = " << HW << ";\n";
    lu << "int DHW = " << DHW << ";\n";
    lu << "int CDHW = " << CDHW << ";\n";
    lu << "int magic_N = " << magic_N << ";\n";
    lu << "int shift_N = " << shift_N << ";\n";
    lu << "int P = " << shape.P << ";\n";
    lu << "int Q = " << shape.Q << ";\n";
    lu << "int magic_P = " << magic_P << ";\n";
    lu << "int shift_P = " << shift_P << ";\n";

    lu << "int PQ = " << PQ << ";\n";
    lu << "int MPQ = " << MPQ << ";\n";
    lu << "int KMPQ = " << KMPQ << ";\n";
    lu << "int S = " << shape.S << ";\n";
    lu << "int RS = " << RS << ";\n";
    lu << "int TRS = " << TRS << ";\n";

    lu << "int magic_S = " << magic_S << ";\n";
    lu << "int shift_S = " << shift_S << ";\n";
    lu << "int magic_RS = " << magic_RS << ";\n";
    lu << "int shift_RS = " << shift_RS << ";\n";

    lu << "int str_d = " << shape.STRIDE_D << ";\n";
    lu << "int str_h = " << shape.STRIDE_H << ";\n";
    lu << "int str_w = " << shape.STRIDE_W << ";\n";
    lu << "int pad_d = " << shape.PAD_D << ";\n";
    lu << "int pad_h = " << shape.PAD_H << ";\n";
    lu << "int pad_w = " << shape.PAD_W << ";\n";
    /*CONST*/

    lu << "const int tid = threadIdx.x;\n";
    lu << "if (tid < 32)\n";
    lu.block_begin();
    {
        lu << "const int q = blockIdx.x;\n";
        lu << "const int mp = blockIdx.y;\n";
        lu << "const int nk = blockIdx.z;\n";
        lu << "const int k = division_by_invariant_multiplication(nk, magic_N, "
              "shift_N);\n";
        lu << "const int n = nk - k * N;\n";
        lu << "const int m = division_by_invariant_multiplication(mp, magic_P, "
              "shift_P);\n";
        lu << "const int p = mp - m * P;\n";
        lu << "out += n*KMPQ + k*MPQ + m*PQ + mad16(p, Q, q);\n";

        // coordinate transform factors from MPQ to DHW
        lu << "int qs = q * str_w - pad_w;\n";
        lu << "int pr = p * str_h - pad_h;\n";
        lu << "int mt = m * str_d - pad_d;\n";

        lu << "int pool_size = ";
        auto pool_size = include_pad ? "TRS" : "0";
        lu << pool_size << ";\n";

        lu << "float sum = 0.0f;\n";
        lu << "float rcp_pool_size = 1.0f;\n";
        // each warp operates on a single pooling window and
        // reduces the contents of the window within the warp
        lu << "for (int trs = tid; trs < TRS; trs += 32)\n";
        lu.block_begin();
        {
            lu << "int t = division_by_invariant_multiplication(trs, magic_RS, "
                  "shift_RS);\n";
            lu << "int rs = mod16(trs, t, RS);\n";
            lu << "int r  = division_by_invariant_multiplication(rs, magic_S, shift_S);\n";
            lu << "int s  = mod16(rs, r, S);\n";

            // coordinate transformation from TRS to DHW
            // via MPQ transform factors above
            lu << "int x = qs + s;\n";
            lu << "int y = pr + r;\n";
            lu << "int z = mt + t;\n";

            // helper to check participating threads
            lu << "bool bounds_x = (x >= 0) && (x < W);\n";
            lu << "bool bounds_y = (y >= 0) && (y < H);\n";
            lu << "bool bounds_z = (z >= 0) && (z < D);\n";
            lu << "bool within_tensor_bounds = bounds_x && bounds_y && bounds_z;\n";

            if (include_pad == false)
            {
                // count the number of (non-padded) elements
                lu << "pool_size += __popc(__ballot_sync(0xffffffff, "
                      "within_tensor_bounds));\n";
            }
            // this will need to change to k->c once
            // feature pooling support is added
            lu << "int idx = n*CDHW + k*DHW + z*HW + y*W + x;\n";
            lu << "sum += load(in,idx,within_tensor_bounds);\n";
        }
        lu.block_end();

        lu << "rcp_pool_size = 1.0f / (float)pool_size;\n";
        // reduce pooling window within warp.
        // this could be improved by calculating the
        // pooling windows each thread can partake in to
        // reduce loads and increase coalescing. in that case,
        // multiple warps per block would be required and the
        // warp reduced sums would need to be accumulated in
        // shared memory
        lu << "for (int i = 16; i > 0; i >>= 1)\n";
        lu.block_begin();
        {
            lu << "sum += __shfl_xor_sync(0xffffffff,sum,i,32);\n";
        }
        lu.block_end();
        // write result to output
        lu << "if (tid == 0)\n";
        lu.block_begin();
        {
            lu << "*out = sum * rcp_pool_size;\n";
        }
        lu.block_end();
    }
    lu.block_end();

    return _lu;
}

void cuda::AvgPool1D::set_launch_config()
{
    m_gridDim = dim3(shape.Q, shape.M * shape.P, shape.N * shape.K);
    m_blockDim = dim3(32, 1, 1);
}

LanguageUnit_p cuda::AvgPool1D::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    _lu->require(header::cuda);
    _lu->require(declaration::division_by_invariant_multiplication);
    _lu->require(declaration::load);
    _lu->require(declaration::mad16);
    _lu->require(declaration::mod16);

    return _lu;
}

cuda::AvgPoolmD::AvgPoolmD(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto avg_pool = static_pointer_cast<nnfusion::op::AvgPool>(ctx->gnode->get_op_ptr());
    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    window_shape = nnfusion::Shape(avg_pool->get_window_shape());
    padding_below = nnfusion::Shape(avg_pool->get_padding_below());
    padding_above = nnfusion::Shape(avg_pool->get_padding_above());
    window_stride = nnfusion::Strides(avg_pool->get_window_movement_strides());
    include_pad = avg_pool->get_include_padding_in_avg_computation();
    input_type = ctx->inputs[0]->get_element_type();
    output_type = ctx->outputs[0]->get_element_type();

    std::stringstream tag;
    tag << "cudnn_avgpool_dtype_" << output_type.c_type_string() << "_i" << join(input_shape, "_")
        << "_o" << join(output_shape, "_") << "_ws" << join(window_shape, "_") << "_wst"
        << join(window_stride, "_") << "_pb" << join(padding_below, "_") << "_pb"
        << join(padding_above, "_");
    custom_tag = tag.str();

    NNFUSION_CHECK(input_shape.size() == 3 || input_shape.size() == 4 || input_shape.size() == 5)
        << "Input shape size of AvgPoolmD is invalid, shape size: " << input_shape.size()
        << "expected 3, 4 or 5";
}

LanguageUnit_p cuda::AvgPoolmD::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    auto rank = input_shape.size();

    auto _input_shape = input_shape;
    auto _output_shape = output_shape;

    if (rank == 3)
    {
        window_shape.insert(window_shape.begin(), 1);
        padding_below.insert(padding_below.begin(), 0);
        window_stride.insert(window_stride.begin(), 1);
        _input_shape.insert(_input_shape.begin() + 2, 1);
        _output_shape.insert(_output_shape.begin() + 2, 1);
        rank = 4;
    }

    auto cudnn_avg_type = include_pad ? "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING"
                                      : "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING";

    auto input_desc = cudnn_tensor_descriptor_from_shape(_input_shape, "input_desc", input_type);
    auto output_desc =
        cudnn_tensor_descriptor_from_shape(_output_shape, "output_desc", output_type);
    lu << input_desc->get_code();
    lu << output_desc->get_code();

    lu << "cudnnPoolingDescriptor_t desc;\n";
    lu << "cudnnCreatePoolingDescriptor(&desc);\n";
    if (rank == 4)
    {
        lu << "CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc,"
           << " " << cudnn_avg_type << ","
           << " CUDNN_NOT_PROPAGATE_NAN," << static_cast<int>(window_shape[0]) << ", "
           << static_cast<int>(window_shape[1]) << ", " << static_cast<int>(padding_below[0])
           << ", " << static_cast<int>(padding_below[1]) << ", "
           << static_cast<int>(window_stride[0]) << ", " << static_cast<int>(window_stride[1])
           << "));\n";
    }
    else /*op->input_shape.size() == 5*/
    {
        std::vector<int> w_strides(window_stride.size());
        std::vector<int> w_shape(window_stride.size());
        std::vector<int> w_padding(padding_below.size());
        for (int i = 0; i < window_shape.size(); i++)
        {
            w_shape[i] = static_cast<int>(window_shape[i]);
            w_strides[i] = static_cast<int>(window_stride[i]);
            w_padding[i] = static_cast<int>(padding_below[i]);
        }

        auto expand_vector_int = [](string name, vector<int>& d) {
            stringstream ss;
            NNFUSION_CHECK(d.size() > 0);
            ss << "int " << name << "[] = {";
            for (int i = 0; i + 1 < d.size(); i++)
                ss << to_string(d[i]) << ", ";
            ss << to_string(d.back()) << "}\n";
            return ss.str();
        };

        lu << expand_vector_int("w_shape", w_shape);
        lu << expand_vector_int("w_strides", w_strides);
        lu << expand_vector_int("w_padding", w_padding);

        lu << "CUDNN_SAFE_CALL(cudnnSetPoolingNdDescriptor(desc, "
           << " " << cudnn_avg_type << ","
           << "CUDNN_NOT_PROPAGATE_NAN, "
           << "3, w_shape, w_padding, w_strides));\n";
    }

    lu << "const float alpha = 1.0;\n";
    lu << "const float beta = 0.0;\n";

    lu << "CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle,"
       << " desc,"
       << " &alpha,"
       << " input_desc,"
       << " input0,"
       << " &beta,"
       << " output_desc,"
       << " output0));\n";

    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));\n";

    return _lu;
}

LanguageUnit_p cuda::AvgPoolmD::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    _lu->require(header::cudnn);
    //_lu->require(declaration::cudnn_handle);
    _lu->require(macro::CUDNN_SAFE_CALL);

    return _lu;
}

LanguageUnit_p cuda::AvgPoolmD::emit_function_signature()
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

LanguageUnit_p cuda::AvgPoolmDGrad::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    _lu->require(header::cudnn);
    //_lu->require(declaration::cudnn_handle);
    _lu->require(macro::CUDNN_SAFE_CALL);

    return _lu;
}

LanguageUnit_p cuda::AvgPoolmDGrad::emit_function_signature()
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

cuda::AvgPoolmDGrad::AvgPoolmDGrad(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto avg_pool = static_pointer_cast<nnfusion::op::AvgPoolBackprop>(ctx->gnode->get_op_ptr());
    // x y dy
    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());
    d_output_shape = nnfusion::Shape(ctx->inputs[2]->get_shape());
    d_input_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

    window_shape = nnfusion::Shape(avg_pool->get_window_shape());
    padding_below = nnfusion::Shape(avg_pool->get_padding_below());
    padding_above = nnfusion::Shape(avg_pool->get_padding_above());
    window_stride = nnfusion::Strides(avg_pool->get_window_movement_strides());
    include_pad = avg_pool->get_include_padding_in_avg_computation();
    input_type = ctx->inputs[0]->get_element_type();
    output_type = ctx->inputs[2]->get_element_type();

    NNFUSION_CHECK(input_shape.size() == 3 || input_shape.size() == 4 || input_shape.size() == 5)
        << "Input shape size of AvgPoolmD is invalid, shape size: " << input_shape.size()
        << "expected 3, 4 or 5";

    std::stringstream tag;
    tag << "cudnn_avgpool_grad_dtype_" << output_type.c_type_string() << "_i"
        << join(input_shape, "_") << "_o" << join(output_shape, "_") << "_ws"
        << join(window_shape, "_") << "_wst" << join(window_stride, "_") << "_pb"
        << join(padding_below, "_") << "_pb" << join(padding_above, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::AvgPoolmDGrad::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    auto rank = input_shape.size();

    auto cudnn_avg_type = include_pad ? "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING"
                                      : "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING";

    auto _input_shape = input_shape;
    auto _d_input_shape = d_input_shape;
    auto _output_shape = output_shape;
    auto _d_output_shape = d_output_shape;

    if (rank == 3)
    {
        window_shape.insert(window_shape.begin(), 1);
        padding_below.insert(padding_below.begin(), 0);
        window_stride.insert(window_stride.begin(), 1);
        _input_shape.insert(_input_shape.begin() + 2, 1);
        _output_shape.insert(_output_shape.begin() + 2, 1);
        _d_input_shape.insert(_d_input_shape.begin() + 2, 1);
        _d_output_shape.insert(_d_output_shape.begin() + 2, 1);
        rank = 4;
    }

    // y dy x dx
    auto input_desc = cudnn_tensor_descriptor_from_shape(_input_shape, "input_desc", input_type);
    auto d_input_desc =
        cudnn_tensor_descriptor_from_shape(_d_input_shape, "d_input_desc", input_type);
    auto output_desc =
        cudnn_tensor_descriptor_from_shape(_output_shape, "output_desc", output_type);
    auto d_output_desc =
        cudnn_tensor_descriptor_from_shape(_d_output_shape, "d_output_desc", output_type);

    lu << "cudnnPoolingDescriptor_t desc;\n";
    lu << "cudnnCreatePoolingDescriptor(&desc);\n";

    lu << input_desc->get_code();
    lu << d_input_desc->get_code();
    lu << output_desc->get_code();
    lu << d_output_desc->get_code();

    if (rank == 4)
    {
        lu << "CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc,"
           << " " << cudnn_avg_type << ","
           << " CUDNN_NOT_PROPAGATE_NAN," << static_cast<int>(window_shape[0]) << ", "
           << static_cast<int>(window_shape[1]) << ", " << static_cast<int>(padding_below[0])
           << ", " << static_cast<int>(padding_below[1]) << ", "
           << static_cast<int>(window_stride[0]) << ", " << static_cast<int>(window_stride[1])
           << "));\n";
    }
    else if (rank == 5)
    {
        std::vector<int> w_strides(window_stride.size());
        std::vector<int> w_shape(window_stride.size());
        std::vector<int> w_padding(padding_below.size());
        for (int i = 0; i < window_shape.size(); i++)
        {
            w_shape[i] = static_cast<int>(window_shape[i]);
            w_strides[i] = static_cast<int>(window_stride[i]);
            w_padding[i] = static_cast<int>(padding_below[i]);
        }

        auto expand_vector_int = [](string name, vector<int>& d) {
            stringstream ss;
            NNFUSION_CHECK(d.size() > 0);
            ss << "int " << name << "[] = {";
            for (int i = 0; i + 1 < d.size(); i++)
                ss << to_string(d[i]) << ", ";
            ss << to_string(d.back()) << "}\n";
            return ss.str();
        };

        lu << expand_vector_int("w_shape", w_shape);
        lu << expand_vector_int("w_strides", w_strides);
        lu << expand_vector_int("w_padding", w_padding);

        lu << "CUDNN_SAFE_CALL(cudnnSetPoolingNdDescriptor(desc, "
           << " " << cudnn_avg_type << ","
           << "CUDNN_NOT_PROPAGATE_NAN, "
           << "3, w_shape, w_padding, w_strides));\n";
    }

    lu << "const float alpha = 1.0;\n";
    lu << "const float beta = 0.0;\n";

    lu << "CUDNN_SAFE_CALL(cudnnPoolingBackward(cudnn_handle,"
       << " desc,"
       << " &alpha,"
       << " output_desc,"
       << " input1,"
       << " d_output_desc,"
       << " input2,"
       << " input_desc,"
       << " input0,"
       << " &beta,"
       << " d_input_desc,"
       << " output0"
       << " ));\n";

    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(d_input_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(d_output_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));\n";

    return _lu;
}

// REGISTER_KERNEL_EMITTER(
//     "AvgPool",                                                                    // op_name
//     Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
//     cuda::AvgPool1D)                                                              // constructor

REGISTER_KERNEL_EMITTER(
    "AvgPool",                                                                     // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cudnn_kernel").Priority(2), // attrs
    cuda::AvgPoolmD)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "AvgPoolBackprop",                                                             // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cudnn_kernel").Priority(2), // attrs
    cuda::AvgPoolmDGrad)                                                           // constructor