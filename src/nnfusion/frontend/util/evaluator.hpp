// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------

#pragma once

#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/cpu/cpu_kernel_emitter.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"
#include "nnfusion/frontend/frontend_base.hpp"
DECLARE_bool(fuse_cpuprofiler);
DECLARE_bool(fantares_mode);
namespace nnfusion
{
    namespace frontend
    {
        namespace
        {
            std::vector<std::vector<char>>
                get_node_outputs(std::shared_ptr<GNode> gnode, int depth = 0, int arg_idx = 0)
            {
                NNFUSION_CHECK(gnode->get_op_type() != "Parameter");
                NNFUSION_LOG(INFO) << "[" << depth << ":" << arg_idx
                                   << "] Working for node: " << gnode->get_name() << ": "
                                   << gnode->get_op_type();
                static std::map<std::shared_ptr<GNode>, std::vector<std::vector<char>>> dict;
                auto it = dict.find(gnode);
                if (it != dict.end())
                    return it->second;

                if (gnode->is_constant())
                {
                    auto const_op = std::dynamic_pointer_cast<op::Constant>(gnode->get_op_ptr());
                    std::vector<char> one(const_op->get_data_size());
                    memcpy(one.data(), const_op->get_data_ptr(), one.size());
                    // for (int i = 0; i < std::min(10LU, one.size()); ++i)
                    //     NNFUSION_LOG(INFO) << one[i];
                    // puts("...");
                    auto& it = dict[gnode];
                    it.push_back(std::move(one));
                    NNFUSION_CHECK(one.size() == 0);
                    return it;
                }

                std::vector<std::vector<char>> _inputs, _outputs;
                int arg_cnt = 0;
                for (auto in_edge : gnode->get_in_edges())
                {
                    auto input_node = in_edge->get_src();
                    auto outs = get_node_outputs(input_node, depth + 1, arg_cnt++);
                    for (auto& out : outs)
                    {
                        _inputs.emplace_back(std::move(out));
                        NNFUSION_CHECK(out.size() == 0);
                    }
                }

                // Prepare runtime backend
                nnfusion::profiler::IProfilingRuntime::Pointer runtime = nullptr;
                std::vector<shared_ptr<const KernelRegistration>> kernel_regs;

                runtime = nnfusion::profiler::RocmDefaultRuntime::Runtime();
                if (FLAGS_fuse_cpuprofiler)
                {
                    runtime = nnfusion::profiler::ReferenceRuntime::Runtime();
                    if (FLAGS_fantares_mode)
                    {
                        shared_ptr<const KernelRegistration> kernel_reg =
                            kernels::Name(gnode->get_op_type())
                                .Device(GENERIC_CPU)
                                .TypeConstraint(element::f32)
                                .Tag("reference")
                                .Priority(0)
                                .KernelFactory([](shared_ptr<kernels::KernelContext> context)
                                                   -> shared_ptr<kernels::KernelEmitter> {
                                    return make_shared<
                                        kernels::cpu::AntaresCpuReferenceKernelEmitter>(context);
                                })
                                .Build();
                        kernel_regs = {kernel_reg};
                        cpu::AntaresCpuReferenceKernelEmitter::codegen_cpu_reference_kernel_sync(
                            gnode);
                    }
                    else
                    {
                        kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                            gnode->get_op_type(), GENERIC_CPU, element::f32);
                    }
                }
                else
                {
                    if (runtime->check_env())
                    {
                        kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                            gnode->get_op_type(), ROCM_GPU, element::f32);
                        if (kernel_regs.size() == 0)
                            kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                                gnode->get_op_type(), CUDA_GPU, element::f32);
                    }
                    else
                    {
                        runtime = nnfusion::profiler::CudaDefaultRuntime::Runtime();
                        NNFUSION_CHECK(runtime->check_env());
                        kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                            gnode->get_op_type(), CUDA_GPU, element::f32);
                    }
                }

                bool const_infer_success = false;
                shared_ptr<KernelContext> ctx(new KernelContext(gnode));
                NNFUSION_LOG(INFO) << "[" << depth << "] Evaluate node: " << gnode->get_name()
                                   << ": " << gnode->get_op_type();
                for (auto& kernel_reg : kernel_regs)
                {
                    auto kernel = kernel_reg->m_factory(ctx);
                    if (!kernel->get_or_emit_source())
                        continue;

                    nnfusion::profiler::ProfilingContext::Pointer pctx =
                        make_shared<nnfusion::profiler::ProfilingContext>(kernel);
                    pctx->warmup_times = 0;
                    pctx->host_times = 1;
                    pctx->runtime_times = 1;

                    nnfusion::profiler::Profiler prof(runtime, pctx);
                    if (!prof.mixed_type_execute(_inputs, _outputs))
                        continue;

                    NNFUSION_LOG(INFO) << "  For node `" << gnode->get_name()
                                       << "`: get runtime output results of size "
                                       << _outputs.size();
                    const_infer_success = true;
                    break;
                }
                return dict[gnode] = _outputs;
            }

            template <typename T, typename VecT = T>
            std::vector<VecT> GetValueFromConstOp(std::shared_ptr<op::Constant> ng_constant_op)
            {
                // the data type of nnfusion::Shape is size_t
                std::vector<VecT> dst_values;
                std::vector<T> values = ng_constant_op->get_vector<T>();
                dst_values.resize(values.size());

                for (size_t i = 0; i < values.size(); i++)
                {
                    dst_values[i] = static_cast<VecT>(values[i]);
                }
                return dst_values;
            }

            template <typename T, typename S>
            void fill_values(std::vector<T>& dst, std::vector<char> src)
            {
                NNFUSION_CHECK(src.size() % sizeof(S) == 0);
                dst.resize(src.size() / sizeof(S));
                S* raw_data = (S*)src.data();
                for (int i = 0; i < dst.size(); ++i)
                    dst[i] = raw_data[i];
            }

            // TODO: currently we only get the first gnode output, might add an out_index argument.
            template <typename T>
            bool GetValueFromNGraphOp(std::shared_ptr<GNode> gnode, std::vector<T>* values)
            {
                if (!gnode->is_constant())
                {
                    auto outs = get_node_outputs(gnode);
                    NNFUSION_CHECK(outs.size() == 1) << "Expect only 1 output for gnode "
                                                     << gnode->get_name() << ", but " << outs.size()
                                                     << " found.";
                    auto out_type = gnode->get_output_element_type(0);
                    NNFUSION_LOG(INFO) << "Asking for Constant value from op-type: "
                                       << gnode->get_op_type();
                    NNFUSION_LOG(INFO) << "Type of Output Value is " << out_type.c_type_string();

                    if (out_type == nnfusion::element::f32)
                        fill_values<T, float>(*values, outs[0]);
                    else if (out_type == nnfusion::element::f64)
                        fill_values<T, double>(*values, outs[0]);
                    else if (out_type == nnfusion::element::i32)
                        fill_values<T, int>(*values, outs[0]);
                    else if (out_type == nnfusion::element::u32)
                        fill_values<T, unsigned>(*values, outs[0]);
                    else if (out_type == nnfusion::element::i64)
                        fill_values<T, int64>(*values, outs[0]);
                    else
                    {
                        NNFUSION_CHECK_FAIL() << "Unsupport op-type conversion, op-type = "
                                              << out_type;
                    }
                    return true;
                }
                auto ng_constant_op = std::dynamic_pointer_cast<op::Constant>(gnode->get_op_ptr());
                auto ng_element_type = gnode->get_output_element_type(0);

                if (sizeof(T) != ng_element_type.size())
                    NNFUSION_LOG(NNFUSION_WARNING) << "Datatypes byte size are not same.";

                if (ng_element_type == nnfusion::element::f32)
                {
                    *values = GetValueFromConstOp<float, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::f16)
                {
                    *values = GetValueFromConstOp<element::half, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::f64)
                {
                    *values = GetValueFromConstOp<double, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::i8)
                {
                    *values = GetValueFromConstOp<int8, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::i16)
                {
                    *values = GetValueFromConstOp<int16, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::i32)
                {
                    *values = GetValueFromConstOp<int32, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::i64)
                {
                    if (ng_element_type.size() == sizeof(int32_t))
                        *values = GetValueFromConstOp<int32_t, T>(ng_constant_op);
                    else
                        *values = GetValueFromConstOp<int64, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::u8)
                {
                    *values = GetValueFromConstOp<uint8, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::u16)
                {
                    *values = GetValueFromConstOp<uint16, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::u32)
                {
                    *values = GetValueFromConstOp<uint32, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::u64)
                {
                    *values = GetValueFromConstOp<uint64, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::boolean)
                {
                    *values = GetValueFromConstOp<int16_t, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::character)
                {
                    *values = GetValueFromConstOp<char, T>(ng_constant_op);
                }
                else
                {
                    return false;
                }
                return true;
            }
        } // namespace
    }     // namespace frontend
} // namespace nnfusion
