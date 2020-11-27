// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DEFINE_bool(frocm_candidate_kernels, true, "Enable some candidate kernels in ROCm.");

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class RocmManualBroadcast : public CudaEmitter
            {
            public:
                RocmManualBroadcast(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                {
                }

                LanguageUnit_p put_source(const std::string& src, bool update_config = false)
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << src;
                    if (update_config)
                    {
                        int at_bx = src.find("// [thread_extent] blockIdx.x = "),
                            blockX = (at_bx >= 0)
                                         ? atoi(src.c_str() + at_bx +
                                                sizeof("// [thread_extent] blockIdx.x = ") - 1)
                                         : 1;
                        int at_by = src.find("// [thread_extent] blockIdx.y = "),
                            blockY = (at_by >= 0)
                                         ? atoi(src.c_str() + at_by +
                                                sizeof("// [thread_extent] blockIdx.y = ") - 1)
                                         : 1;
                        int at_bz = src.find("// [thread_extent] blockIdx.z = "),
                            blockZ = (at_bz >= 0)
                                         ? atoi(src.c_str() + at_bz +
                                                sizeof("// [thread_extent] blockIdx.z = ") - 1)
                                         : 1;
                        int at_tx = src.find("// [thread_extent] threadIdx.x = "),
                            threadX = (at_tx >= 0)
                                          ? atoi(src.c_str() + at_tx +
                                                 sizeof("// [thread_extent] threadIdx.x = ") - 1)
                                          : 1;
                        int at_ty = src.find("// [thread_extent] threadIdx.y = "),
                            threadY = (at_ty >= 0)
                                          ? atoi(src.c_str() + at_ty +
                                                 sizeof("// [thread_extent] threadIdx.y = ") - 1)
                                          : 1;
                        int at_tz = src.find("// [thread_extent] threadIdx.z = "),
                            threadZ = (at_tz >= 0)
                                          ? atoi(src.c_str() + at_tz +
                                                 sizeof("// [thread_extent] threadIdx.z = ") - 1)
                                          : 1;

                        m_gridDim = dim3(blockX, blockY, blockZ);
                        m_blockDim = dim3(threadX, threadY, threadZ);
                    }
                    return _lu;
                }

                LanguageUnit_p emit_function_body() override
                {
                    bool using_fixed = FLAGS_frocm_candidate_kernels;
                    if (!using_fixed)
                        return nullptr;

                    auto& ctx = m_context;
                    auto input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    auto output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

                    auto node =
                        static_pointer_cast<nnfusion::op::Broadcast>(ctx->gnode->get_op_ptr());
                    auto axes = node->get_broadcast_axes();

#if 0 // for Debug
                    for (auto &it: input_shape) printf("%d, ", (int)it); puts("");
                    for (auto &it: output_shape) printf("%d, ", (int)it); puts("");
                    for (auto &it: axes) printf("%d, ", (int)it); puts("");
                    puts("====================");
#endif
                    if (input_shape == nnfusion::Shape({6, 512, 512}) &&
                        output_shape == nnfusion::Shape({6, 16, 512, 512}))
                    {
                        std::string src = R"(
  for (int i_inner_inner_inner = 0; i_inner_inner_inner < 6; ++i_inner_inner_inner) {
  // [thread_extent] blockIdx.y = 1
  // [thread_extent] threadIdx.y = 1
    for (int j_inner_inner_inner = 0; j_inner_inner_inner < 8; ++j_inner_inner_inner) {
  // [thread_extent] blockIdx.z = 1024
  // [thread_extent] threadIdx.z = 32
      for (int k_inner_inner_inner = 0; k_inner_inner_inner < 2; ++k_inner_inner_inner) {
        output0[(((((i_inner_inner_inner * 4194304) + (j_inner_inner_inner * 262144)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner)] = input0[((((i_inner_inner_inner * 262144) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner)];
        output0[((((((i_inner_inner_inner * 4194304) + (j_inner_inner_inner * 262144)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 2097152)] = input0[((((i_inner_inner_inner * 262144) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner)];
        output0[((((((i_inner_inner_inner * 4194304) + (j_inner_inner_inner * 262144)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 64)] = input0[(((((i_inner_inner_inner * 262144) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 64)];
        output0[((((((i_inner_inner_inner * 4194304) + (j_inner_inner_inner * 262144)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 2097216)] = input0[(((((i_inner_inner_inner * 262144) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 64)];
        output0[((((((i_inner_inner_inner * 4194304) + (j_inner_inner_inner * 262144)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 128)] = input0[(((((i_inner_inner_inner * 262144) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 128)];
        output0[((((((i_inner_inner_inner * 4194304) + (j_inner_inner_inner * 262144)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 2097280)] = input0[(((((i_inner_inner_inner * 262144) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 128)];
        output0[((((((i_inner_inner_inner * 4194304) + (j_inner_inner_inner * 262144)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 192)] = input0[(((((i_inner_inner_inner * 262144) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 192)];
        output0[((((((i_inner_inner_inner * 4194304) + (j_inner_inner_inner * 262144)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 2097344)] = input0[(((((i_inner_inner_inner * 262144) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 2)) + k_inner_inner_inner) + 192)];
      }
    }
  }
)";
                        return put_source(src, true);
                    }

                    // (1, ..) -> (1, ..) to (..) -> (..)
                    if (input_shape.size() > 1 && output_shape.size() > 1 && input_shape[0] == 1 &&
                        output_shape[0] == 1 && !axes.count(0))
                    {
                        auto input_shape2 = nnfusion::Shape(), output_shape2 = nnfusion::Shape();
                        auto axes2 = nnfusion::AxisSet();
                        for (int i = 1; i < input_shape.size(); ++i)
                            input_shape2.push_back(input_shape[i]);
                        for (int i = 1; i < output_shape.size(); ++i)
                            output_shape2.push_back(output_shape[i]);
                        for (auto it : axes)
                            axes2.insert(it - 1);
                        input_shape = std::move(input_shape2);
                        output_shape = std::move(output_shape2);
                        axes = std::move(axes2);
                    }

                    // (X, ..) -> (U, X, ..) to (V) -> (U, V)
                    if (input_shape.size() > 1 && input_shape.size() < output_shape.size())
                    {
                        int diff = output_shape.size() - input_shape.size();
                        bool tail = axes.count(0) && axes.size() == 1;
                        size_t tail_size = 1;
                        for (int i = 0; tail && i < input_shape.size(); ++i)
                        {
                            if (input_shape[i] != output_shape[diff + i])
                                tail = false;
                            tail_size *= input_shape[i];
                        }
                        if (tail)
                        {
                            input_shape.resize(0);
                            output_shape.resize(diff);
                            input_shape.push_back(tail_size);
                            output_shape.push_back(tail_size);
                        }
                    }

                    if (m_context->dtypes[0] != "float" || m_context->dtypes[1] != "float")
                    {
                        return nullptr;
                    }

                    size_t in_size = 1, out_size = 1;
                    for (auto& it : input_shape)
                        in_size *= it;
                    for (auto& it : output_shape)
                        out_size *= it;

                    std::string code;
                    if (in_size == out_size)
                    {
                        // DtoD_copy
                        return nullptr;
                    }
                    else if (in_size == 1)
                    {
                        // DtoD_memset
                        int threads;
                        if (out_size % 4096 == 0)
                            threads = 1024;
                        else if (out_size % 2048 == 0)
                            threads = 512;
                        else if (out_size % 1024 == 0)
                            threads = 256;
                        else if (out_size % 512 == 0)
                            threads = 128;
                        else if (out_size % 256 == 0)
                            threads = 64;
                        else
                            return nullptr;

                        m_gridDim = dim3(out_size / threads / 4, 1, 1);
                        m_blockDim = dim3(threads, 1, 1);

                        code = nnfusion::op::create_code_from_template(
                            R"(
		((float4*)output0)[blockIdx.x * @blockDim_x@ + threadIdx.x] = make_float4(*input0, *input0, *input0, *input0);
	)",
                            {
                                {"blockDim_x", threads},
                            });
                    }
                    else if ((input_shape.size() == 1 && output_shape.size() == 2 &&
                              axes.count(0) > 0 && axes.size() == 1))
                    {
                        // (1, B) to (A, B)
                        NNFUSION_CHECK(input_shape[0] == output_shape[1]);
                        int blocks, blocks2 = 1, threads;
                        std::string vec_type = "float4";

                        if (input_shape[0] == 4096)
                            threads = 1024, blocks = output_shape[0];
                        else if (input_shape[0] == 1024)
                            threads = 256, blocks = output_shape[0];
                        else if (input_shape[0] == 512)
                            threads = 128, blocks = output_shape[0];
                        else if (input_shape[0] >= 1000 && input_shape[0] < 1024)
                            threads = input_shape[0], blocks = output_shape[0], vec_type = "float";
                        else if (input_shape[0] % 4096 == 0)
                            threads = 1024, blocks = output_shape[0],
                            blocks2 = input_shape[0] / 4096;
                        else
                            return nullptr;

                        m_gridDim = dim3(blocks, blocks2, 1);
                        m_blockDim = dim3(threads, 1, 1);

                        if (blocks2 == 1)
                        {
                            code = nnfusion::op::create_code_from_template(
                                R"(
			((@vec_type@*)output0)[blockIdx.x * @blockDim_x@ + threadIdx.x] = ((@vec_type@*)input0)[threadIdx.x];
		)",
                                {{"blockDim_x", threads}, {"vec_type", vec_type}});
                        }
                        else
                        {
                            code = nnfusion::op::create_code_from_template(
                                R"(
			((@vec_type@*)output0)[blockIdx.x * @blockDim_x@ * @blockDim_y@ + blockIdx.y * @blockDim_x@ + threadIdx.x] = ((@vec_type@*)input0)[blockIdx.y * @blockDim_x@ + threadIdx.x];
		)",
                                {{"blockDim_x", threads},
                                 {"blockDim_y", blocks2},
                                 {"vec_type", vec_type}});
                        }
                    }
                    else if ((input_shape.size() == 1 && output_shape.size() == 2 &&
                              axes.count(1) > 0 && axes.size() == 1))
                    {
                        // (A, 1) to (A, B)
                        NNFUSION_CHECK(input_shape[0] == output_shape[0]);

                        int blocks, threads;
                        if (output_shape[1] == 4096)
                            threads = 1024, blocks = output_shape[0];
                        else if (output_shape[1] == 2048)
                            threads = 512, blocks = output_shape[0];
                        else if (output_shape[1] == 1024)
                            threads = 256, blocks = output_shape[0];
                        else if (output_shape[1] == 512)
                            threads = 128, blocks = output_shape[0];
                        else
                            return nullptr;

                        m_gridDim = dim3(blocks, 1, 1);
                        m_blockDim = dim3(threads, 1, 1);

                        code = nnfusion::op::create_code_from_template(
                            R"(
		((float4*)output0)[blockIdx.x * @blockDim_x@ + threadIdx.x] = make_float4(input0[blockIdx.x], input0[blockIdx.x], input0[blockIdx.x], input0[blockIdx.x]);
	)",
                            {{"blockDim_x", threads}});
                    }
                    else
                        return nullptr;

                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << code << "\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override {}
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register Pad kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Broadcast",                                               // op_name
                        Device(ROCM_GPU).TypeConstraint(element::f32).Priority(4), // attrs
                        cuda::RocmManualBroadcast)                                 // constructor
