// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "cuda_helper.hpp"
#include "cuda_langunit.hpp"
#include "nnfusion/core/kernels/antares_ke_imp.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/engine/async_manager.hpp"

DECLARE_bool(fantares_mode);

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            struct dim3
            {
                dim3()
                    : x(1)
                    , y(1)
                    , z(1)
                {
                }
                dim3(int x, int y, int z)
                    : x(x)
                    , y(y)
                    , z(z)
                {
                }
                int x, y, z;
            };

            class CudaEmitter : public KernelEmitter
            {
            public:
                CudaEmitter(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cuda")
                {
                }

                virtual bool is_static_function() override { return false; }
                // Need to regenerate function call with new assigned launch config(stream).
                LanguageUnit_p emit_function_call() override;

                dim3 get_grid_dim() { return m_gridDim; }
                dim3 get_block_dim() { return m_blockDim; }
                virtual shared_ptr<nnfusion::cache::KernelEntry> get_kernel_cache_entry(
                    shared_ptr<nnfusion::cache::KernelEntry> kernel_entry = nullptr) override;

            protected:
                // config the blockDim and gridDim
                virtual void set_launch_config() = 0;

                LanguageUnit_p emit_function_signature() override;

                virtual void emit_thread_sync(LanguageUnit& lu) { lu << "__syncthreads();\n"; }
                virtual void emit_alloc_shared(LanguageUnit& lu,
                                               std::string symbol,
                                               std::string type,
                                               size_t size)
                {
                    lu << "__shared__ " << type << " " << symbol << "[" << size << "];\n";
                }

                dim3 m_blockDim;
                dim3 m_gridDim;
            };

            class BlockCudaEmitter : public CudaEmitter
            {
            public:
                BlockCudaEmitter(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                    , num_local_thread_sync(0)
                    , shared_memory_size(0)
                    , is_emitting_block_kernel(false)
                {
                    shared_memory_log.symbol.clear();
                    shared_memory_log.dtype.clear();
                    shared_memory_log.size.clear();
                }

                static const std::unordered_map<std::string, size_t> size_of_str_type;

                size_t get_shared_memory_size() { return shared_memory_size; }
                FunctionUnit_p get_or_emit_source(bool emit_func_call = false) override
                {
                    if (!m_is_emitted)
                    {
                        KernelEmitter::get_or_emit_source();
                        bool temp = is_emitting_block_kernel;
                        is_emitting_block_kernel = true;
                        m_block_function_unit = this->emit_source();
                        is_emitting_block_kernel = temp;
                    }
                    else
                    {
                        if (emit_func_call)
                            m_function_unit->call_unit = emit_function_call();
                    }
                    return is_emitting_block_kernel ? m_block_function_unit : m_function_unit;
                }

                LanguageUnit_p emit_block_kernel()
                {
                    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel"));
                    auto& lu = *_lu;

                    is_emitting_block_kernel = true;
                    FunctionUnit_p fu = this->get_or_emit_source();
                    is_emitting_block_kernel = false;
                    lu << fu->comment_unit->get_code();
                    lu << this->emit_device_function_signature()->get_code() << "\n";
                    lu.block_begin();
                    lu << this->emit_device_function_body()->get_code();
                    lu.block_end();

                    return _lu;
                }

                LanguageUnit_p emit_device_function_signature();
                LanguageUnit_p emit_device_function_body();

                // this API can only be used inner the function body
                void emit_thread_sync(LanguageUnit& lu) override
                {
                    CudaEmitter::emit_thread_sync(lu);
                    num_local_thread_sync++;
                }

                void emit_alloc_shared(LanguageUnit& lu,
                                       std::string symbol,
                                       std::string type,
                                       size_t size) override
                {
                    if (is_emitting_block_kernel)
                    {
                        lu << type << "* " << symbol << " = (" << type << "*)(shared_buffer + "
                           << shared_memory_size << ");\n";
                        auto iter = size_of_str_type.find(type);
                        NNFUSION_CHECK(iter != size_of_str_type.end()) << "Unknown data type: "
                                                                       << type;
                        shared_memory_size += size * iter->second;

                        shared_memory_log.symbol.push_back(symbol);
                        shared_memory_log.dtype.push_back(type);
                        shared_memory_log.size.push_back(size);
                    }
                    else
                    {
                        CudaEmitter::emit_alloc_shared(lu, symbol, type, size);
                    }
                }

                virtual shared_ptr<nnfusion::cache::KernelEntry> get_kernel_cache_entry(
                    shared_ptr<nnfusion::cache::KernelEntry> kernel_entry = nullptr) override;

            public:
                struct SharedMemoryLog
                {
                    std::vector<std::string> symbol;
                    std::vector<std::string> dtype;
                    std::vector<size_t> size;
                };

            protected:
                size_t num_local_thread_sync;
                SharedMemoryLog shared_memory_log;
                size_t shared_memory_size;
                bool is_emitting_block_kernel;
                FunctionUnit_p m_block_function_unit;
            };

            class CudaElementwiseEmitter : public BlockCudaEmitter
            {
            public:
                CudaElementwiseEmitter(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                {
                }

                virtual std::pair<std::string, shared_ptr<LanguageUnit>> get_op_kernel() = 0;
            };

            class CudaLibEmitter : public KernelEmitter
            {
            public:
                CudaLibEmitter(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cuda_lib")
                {
                }
                virtual bool require_cudnn_handle() { return false; }
                virtual bool require_cublas_handle() { return false; }
            };

            class AntaresCudaKernelEmitter : public BlockCudaEmitter
            {
            public:
                AntaresCudaKernelEmitter(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , m_antares_ke_imp(new AntaresKEImp)
                {
                    GENERIC_OP_LOGGING();
                    if (FLAGS_fantares_mode)
                    {
                        // NNFUSION_LOG(INFO) << "Translate for " << ctx->gnode->get_op_type();

                        ir = nnfusion::op::get_translation(ctx->gnode);
#if 0
                        std::unordered_set<std::string> wl = {
                          "Add", "ApplyGradient", "AvgPool", "BatchMatMul", "Broadcast", "Concat", "Convert", "Convolution", "DepthToSpace", "DepthwiseConv2dNative",
                          "Dot", "Elementwise", "GatherV2", "MaxPool", "OneHot", "Pad", "Relu", "Reshape", "Tile", "Reverse", "Shape", "Slice", "Sum",
                        };
                        if (!ir.empty() && wl.count(ctx->gnode->get_op_type()))
#else
                        if (!ir.empty())
#endif
                        {
                            auto info = m_antares_ke_imp->autogen(ir);
                            if (info.first == "")
                            {
                                NNFUSION_LOG(INFO) << "No Antares codegen for IR: " << ir;
                                return;
                            }
                            { // check output_shapes of NNFusion and Antares, fallback when shapes mismatch.
                                // cout << ctx->gnode->get_op_type() << endl;
                                // cout << ir << endl;
                                auto antares_output_shapes =
                                    AntaresKEImp::get_output_shapes(info.first);
                                bool flag_match = true;

                                if (antares_output_shapes.size() == 0)
                                {
                                    flag_match = false;
                                }

                                if (antares_output_shapes.size() != ctx->outputs.size())
                                {
                                    flag_match = false;
                                }
                                else
                                {
                                    for (int i = 0; i < antares_output_shapes.size(); i++)
                                    {
                                        if (simplify(antares_output_shapes[i]) !=
                                            simplify(ctx->outputs[i]->get_shape()))
                                        {
                                            NNFUSION_LOG(INFO)
                                                << "NNFusion shape: "
                                                << ctx->outputs[i]->get_shape()
                                                << ", Antares shape: " << antares_output_shapes[i];
                                            NNFUSION_LOG(INFO) << info.first;
                                            flag_match = false;
                                            break;
                                        }
                                    }
                                }

                                if (!flag_match)
                                {
                                    NNFUSION_LOG(ERROR)
                                        << "Infershapes of gnode " << ctx->gnode->get_name()
                                        << " do not match in NNFusion and Antares, fallback ";
                                    return;
                                }
                            }

                            antares_code = info.first;
                            m_is_tuned = info.second;

                            std::string annotation = nnfusion::op::get_annotation(ir);
                            // if is_memcpy, no need to request antares server
                            if (annotation.find("|memcpy|") != string::npos)
                            {
                                is_memcpy = true;
                            }
                        }
                        if (ir.empty())
                        {
                            static std::unordered_set<std::string> log_cache;
                            if (log_cache.count(ctx->gnode->get_op_type()) == 0)
                            {
                                NNFUSION_LOG(INFO) << "No Antares Translation for Op: "
                                                   << ctx->gnode->get_op_type();
                                log_cache.insert(ctx->gnode->get_op_type());
                            }
                        }

                        kernel_info =
                            nnfusion::kernels::AntaresKEImp::get_kernel_info(antares_code);
                        NNFUSION_CHECK(!kernel_info.empty());
                        process_antares_kernel_info();
                    }
                }

                virtual shared_ptr<nnfusion::cache::KernelEntry> get_kernel_cache_entry(
                    shared_ptr<nnfusion::cache::KernelEntry> kernel_entry = nullptr) override;

                bool is_eliminative() override;
                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_function_call() override;
                LanguageUnit_p emit_function_signature() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_comments() override
                {
                    auto lu = KernelEmitter::emit_comments();
                    (*lu) << "// IR:\n//" << ir << "\n";
                    return lu;
                };
                void set_launch_config() override {}
                AntaresKEImp::Pointer m_antares_ke_imp;
                std::string antares_code;
                std::string ir;
                bool is_memcpy = false;
                // remove non-1 elements in a shape
                nnfusion::Shape simplify(nnfusion::Shape s)
                {
                    nnfusion::Shape sim;
                    for (auto i : s)
                        if (i != 1)
                            sim.push_back(i);
                    return sim;
                }

            protected:
                // map tensor names and allocate tmp tensor
                void process_antares_kernel_info();
                void find_launch_config(const std::string& str,
                                        dim3& gridDim,
                                        dim3& blockDim,
                                        std::string& block_num);
                std::vector<AntaresKernelInfo::Pointer> kernel_info;
                std::unordered_map<std::string, std::string>
                    tensor_name_map; // antares tensor name : kernel tensor name
                std::map<std::string, std::string> symbol_expr;
                std::string m_blockNum;
            };

            class CacheCudaEmitter : public CudaEmitter
            {
            public:
                CacheCudaEmitter(shared_ptr<KernelContext> ctx,
                                 nnfusion::cache::KernelEntry_p kernel_entry_p)
                    : CudaEmitter(ctx)
                {
                    NNFUSION_CHECK_NOT_NULLPTR(kernel_entry_p);
                    kernel_entry = *kernel_entry_p;

                    NNFUSION_CHECK(!kernel_entry.function.is_null());
                }

            private:
                LanguageUnit_p emit_function_signature() override;
                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                nnfusion::cache::KernelEntry kernel_entry;
            };

            class CacheBlockCudaEmitter : public BlockCudaEmitter
            {
            public:
                CacheBlockCudaEmitter(shared_ptr<KernelContext> ctx,
                                      nnfusion::cache::KernelEntry_p kernel_entry_p)
                    : BlockCudaEmitter(ctx)
                {
                    NNFUSION_CHECK_NOT_NULLPTR(kernel_entry_p);
                    kernel_entry = *kernel_entry_p;

                    NNFUSION_CHECK(!kernel_entry.function.is_null());
                }

            private:
                LanguageUnit_p emit_function_signature() override;
                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                nnfusion::cache::KernelEntry kernel_entry;
            };

            class FusionCudaEmitter : public CudaEmitter
            {
            public:
                FusionCudaEmitter(shared_ptr<KernelContext> ctx, json fusion_group);

            private:
                LanguageUnit_p emit_function_signature() override;
                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                json m_fusion_group;
                string m_code;
                LanguageUnit_p m_body_unitp, m_sig_unitp, m_dep_unitp;
            };

            class CustomCudaKernelEmitter : public BlockCudaEmitter
            {
            public:
                CustomCudaKernelEmitter(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                {
                    auto op_config =
                        nnfusion::op::lookup_op_config(m_context->gnode->get_op_type());
                    std::vector<int32_t> config = op_config.get("launch_config");
                    NNFUSION_CHECK(config.size() == 6);
                    m_gridDim = dim3(config[0], config[1], config[2]);
                    m_blockDim = dim3(config[3], config[4], config[5]);
                    is_memcpy = op_config.get("is_memcpy");
                }

                virtual shared_ptr<nnfusion::cache::KernelEntry> get_kernel_cache_entry(
                    shared_ptr<nnfusion::cache::KernelEntry> kernel_entry = nullptr) override
                {
                    if (kernel_entry == nullptr)
                    {
                        kernel_entry = std::make_shared<nnfusion::cache::KernelEntry>();
                    }
                    if (kernel_entry->source == "")
                    {
                        kernel_entry->source = "Custom";
                    }
                    return BlockCudaEmitter::get_kernel_cache_entry(kernel_entry);
                }

                bool is_eliminative() override
                {
                    return (is_memcpy &&
                            m_context->inputs[0]->is_same_address(m_context->outputs[0]));
                }
                LanguageUnit_p emit_function_body() override
                {
                    std::string body_code;
                    auto op_config =
                        nnfusion::op::lookup_op_config(m_context->gnode->get_op_type());
                    if (op_config.f_kernel_funcs.count("CUDA_GPU") > 0)
                        body_code = op_config.f_kernel_funcs["CUDA_GPU"](m_context->gnode);
                    if (body_code == "")
                        return nullptr;
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu.block_begin();
                    lu << body_code << "\n";
                    lu.block_end();
                    return _lu;
                }
                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }
                void set_launch_config() override {}
                bool is_memcpy = false;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
