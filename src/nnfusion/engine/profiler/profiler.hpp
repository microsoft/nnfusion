// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * \brief Use this Profiler to run each operator
 * \author wenxh
 * \todo This profiler only support linux since it will invoke native commands.
 */
#pragma once

#include <algorithm>
#include <string>

#include "cpu_runtime.hpp"
#include "cuda_runtime.hpp"
#include "nnfusion/common/type/data_buffer.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "profiling_runtime.hpp"
#include "rocm_runtime.hpp"

//Support Linux for now.
#include <dlfcn.h>
#define CLOSE_LIBRARY(a) dlclose(a)
#define DLSYM(a, b) dlsym(a, b)
#define DLIB_SUFFIX ".so"
#define DL_HANDLE void*

using namespace std;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;
using nnfusion::DataBuffer;

namespace nnfusion
{
    namespace profiler
    {
        IProfilingRuntime::Pointer get_default_runtime(NNFusion_DeviceType dev_t);
        IProfilingRuntime::Pointer get_default_runtime(string dev_str);

        ///\brief Profiler will profile a operator or a subgraph. This Profiler class should be treated as interface for Host.
        //Profiler will use the Runtime to run the subject.
        ///\todo To support a subgraph
        class Profiler
        {
        public:
            Profiler(IProfilingRuntime::Pointer rt, ProfilingContext::Pointer context);
            bool execute();
            bool find_best();
            bool execute_all();
            double execute(void** input, void** output);

            ///\brief T should be basic date type: int, float, double;
            template <typename T>
            vector<vector<T>> execute(const vector<vector<T>>& inputs)
            {
                auto& kernel_mem = pctx->kernel_memory;
                kernel_mem->load_inputs(inputs);

                if (rt->execute(pctx, kernel_mem->unsafe_inputs(), kernel_mem->unsafe_outputs()) <
                    0)
                {
                    NNFUSION_LOG(ERROR) << "Failed to execute the kernel.";
                    return vector<vector<T>>();
                }

                return kernel_mem->save_outputs<T>();
            }

            vector<DataBuffer> execute(const vector<DataBuffer>& inputs, element::Type type)
            {
                auto& kernel_mem = pctx->kernel_memory;
                kernel_mem->load_inputs(inputs);

                if (rt->execute(pctx, kernel_mem->unsafe_inputs(), kernel_mem->unsafe_outputs()) <
                    0)
                {
                    NNFUSION_LOG(ERROR) << "Failed to execute the kernel.";
                    return vector<DataBuffer>();
                }

                return kernel_mem->save_outputs(type);
            }

            // multiple inputs (or outputs) may have different element types
            bool mixed_type_execute(const vector<vector<char>>& inputs,
                                    vector<vector<char>>& outputs)
            {
                auto& kernel_mem = pctx->kernel_memory;
                auto kctx = pctx->kernel->m_context;
                NNFUSION_CHECK(inputs.size() == kctx->inputs.size());

                for (size_t i = 0; i < kctx->inputs.size(); i++)
                {
                    auto& t = kctx->inputs[i];
                    size_t _size = t->size();
                    NNFUSION_CHECK(inputs[i].size() == _size);

                    kernel_mem->load_input_from(i, inputs[i].data(), _size);
                }

                if (rt->execute(pctx, kernel_mem->unsafe_inputs(), kernel_mem->unsafe_outputs()) <
                    0)
                {
                    NNFUSION_LOG(ERROR) << "Failed execute the kernel.";
                    return false;
                }

                outputs.clear();
                void** ptrs = kernel_mem->unsafe_outputs();
                for (size_t i = 0; i < kctx->outputs.size(); ++i)
                {
                    auto& t = kctx->outputs[i];
                    size_t _size = t->size();

                    NNFUSION_CHECK(ptrs[i] != nullptr);
                    vector<char> output(_size);
                    memcpy(output.data(), ptrs[i], _size);

                    outputs.push_back(move(output));
                }
                return true;
            }

            ///\brief simple interface for execute
            template <typename T>
            vector<vector<T>> unsafe_execute(const void* val)
            {
                auto& kernel_mem = pctx->kernel_memory;

                size_t offset = 0;
                auto kctx = pctx->kernel->m_context;
                for (size_t i = 0; i < kctx->inputs.size(); i++)
                {
                    auto& t = kctx->inputs[i];
                    size_t _size = t->size();
                    void* newval = (void*)((char*)val + offset);
                    kernel_mem->load_input_from(i, newval, _size);
                    offset += _size;
                }

                if (rt->execute(pctx, kernel_mem->unsafe_inputs(), kernel_mem->unsafe_outputs()) <
                    0)
                {
                    NNFUSION_LOG(ERROR) << "Failed execute the kernel.";
                    return vector<vector<T>>();
                }

                return kernel_mem->save_outputs<T>();
            }

            vector<DataBuffer> unsafe_execute(const void* val, element::Type type)
            {
                auto& kernel_mem = pctx->kernel_memory;

                size_t offset = 0;
                auto kctx = pctx->kernel->m_context;
                for (size_t i = 0; i < kctx->inputs.size(); i++)
                {
                    auto& t = kctx->inputs[i];
                    size_t _size = t->size();
                    void* newval = (void*)((char*)val + offset);
                    kernel_mem->load_input_from(i, newval, _size);
                    offset += _size;
                }

                if (rt->execute(pctx, kernel_mem->unsafe_inputs(), kernel_mem->unsafe_outputs()) <
                    0)
                {
                    NNFUSION_LOG(ERROR) << "Failed execute the kernel.";
                    return vector<DataBuffer>();
                }

                return kernel_mem->save_outputs(type);
            }

            // HOST TENSOR Operations
            ///\brief Allocate spaces for output tensors, but tensors need to be same type.
            /*
            template <typename T>
            vector<vector<T>> allocate_outputs()
            {
                auto& kctx = pctx->kernel->m_context;
                vector<vector<T>> res;
                for (int i = 0; i < kctx->outputs.size(); i++)
                {
                    res.push_back(vector<T>());
                    res[i].resize(kctx->outputs[i]->size(false));
                }
                return move(res);
            }

            template <class T>
            std::vector<T> create_vector(T* t, size_t size)
            {
                std::vector<T> vec;
                for (int i = 0; i < size; i++)
                    vec.push_back(t[i]);
                return vec;
            }

            template <class T>
            void* create_empty_tensor(size_t size)
            {
                T* t = new T[size];
                memset(t, 0, sizeof(T) * size);
                return t;
            }

            template <class T>
            void* create_zeros_tensor(size_t size)
            {
                T* t = new T[size];
                for (size_t i = 0; i < size; i++)
                    t[i] = 1;
                return t;
            }

            template <class T>
            void* create_tensor(T* t, size_t size)
            {
                return t;
            }

            template <class T>
            void* create_tensor(std::vector<T> data)
            {
                T* t = new T[data.size()];
                for (int i = 0; i < data.size(); i++)
                    t[i] = data[i];
                return t;
            }
            */

        private:
            ProfilingContext::Pointer pctx;
            IProfilingRuntime::Pointer rt;
        };

        ///\brief Evaluation for (sub)graph, the subgraph should have none undetermined input.
        class GraphEvaluate
        {
        public:
            GraphEvaluate(shared_ptr<nnfusion::graph::Graph> graph, NNFusion_DeviceType dev_t)
                : gctx(GraphEvaluationContext(graph))
                , dev_type(dev_t)
            {
                rt = get_default_runtime(dev_t);
            }
            template <typename T, typename T1>
            unordered_map<string, vector<vector<T1>>> eval(const vector<vector<T>>& inputs)
            {
                auto parameters = gctx.graph->get_parameters();

                NNFUSION_CHECK(inputs.size() == parameters.size())
                    << "The input size does not match graph's Parameter count";
                for (size_t i = 0; i < parameters.size(); i++)
                {
                    parameter_map[parameters[i]] = i;
                }
                auto ordered_ops = gctx.graph->get_ordered_ops();
                for (auto& op : ordered_ops)
                {
                    create_profiling_contexts(op);
                }

                for (auto& op : ordered_ops)
                {
                    connect_nodes(op, inputs);
                }

                int i = 0;
                for (auto& node : ordered_ops)
                {
                    if (node->get_op_ptr()->is_tensor_op())
                    {
                        continue;
                    }
                    auto pctx = gctx.get_profiling_context(node);
                    // Ensure only run once
                    pctx->warmup_times = 0;
                    pctx->runtime_times = 1;

                    rt->execute(pctx,
                                pctx->kernel_memory->unsafe_inputs(),
                                pctx->kernel_memory->unsafe_outputs());
                }

                unordered_map<string, vector<vector<T1>>> result;
                for (auto& outnode : gctx.graph->get_outputs())
                {
                    if (outnode->is_parameter())
                    {
                        auto vec = inputs[parameter_map[outnode]];
                        result[outnode->get_unique_name()].emplace_back(vec.begin(), vec.end());
                    }
                    ///\todo tensor op?
                    else if (outnode->is_constant())
                    {
                        auto const_node = static_pointer_cast<op::Constant>(outnode->get_op_ptr());
                        NNFUSION_LOG(NNFUSION_WARNING) << "GraphEvaluate::eval might return "
                                                          "unexpected result on constant node, "
                                                          "please use mixed_type_eval instead.";
                        ///\warning const->get_vector<T> only reinterprete memory to desired T.
                        result[outnode->get_unique_name()].push_back(
                            move(const_node->get_vector<T1>()));
                    }
                    else
                    {
                        auto pctx = gctx.get_profiling_context(outnode);
                        result[outnode->get_unique_name()] =
                            pctx->kernel_memory->save_outputs<T1>();
                    }
                }

                // The result data ptr is like result["nodename"]->kernel_memory->unsafe_output(0);
                return move(result);
            }

            unordered_map<string, vector<DataBuffer>> eval(const vector<DataBuffer>& inputs,
                                                           element::Type type_out,
                                                           element::Type type_in)
            {
                auto parameters = gctx.graph->get_parameters();

                NNFUSION_CHECK(inputs.size() == parameters.size())
                    << "The input size does not match graph's Parameter count";
                for (size_t i = 0; i < parameters.size(); i++)
                {
                    parameter_map[parameters[i]] = i;
                }
                auto ordered_ops = gctx.graph->get_ordered_ops();
                for (auto& op : ordered_ops)
                {
                    create_profiling_contexts(op);
                }

                for (auto& op : ordered_ops)
                {
                    connect_nodes(op, inputs);
                }

                int i = 0;
                for (auto& node : ordered_ops)
                {
                    if (node->get_op_ptr()->is_tensor_op())
                    {
                        continue;
                    }
                    auto pctx = gctx.get_profiling_context(node);
                    // Ensure only run once
                    pctx->warmup_times = 0;
                    pctx->runtime_times = 1;

                    rt->execute(pctx,
                                pctx->kernel_memory->unsafe_inputs(),
                                pctx->kernel_memory->unsafe_outputs());
                }

                unordered_map<string, vector<DataBuffer>> result;
                for (auto& outnode : gctx.graph->get_outputs())
                {
                    if (outnode->is_parameter())
                    {
                        DataBuffer vec = std::move(inputs[parameter_map[outnode]]);
                        result[outnode->get_unique_name()].push_back(std::move(vec));
                    }
                    ///\todo tensor op?
                    else if (outnode->is_constant())
                    {
                        auto const_node = static_pointer_cast<op::Constant>(outnode->get_op_ptr());
                        NNFUSION_LOG(NNFUSION_WARNING) << "GraphEvaluate::eval might return "
                                                          "unexpected result on constant node, "
                                                          "please use mixed_type_eval instead.";
                        ///\warning const->get_vector<T> only reinterprete memory to desired T.
                        result[outnode->get_unique_name()].push_back(
                            move(const_node->get_buffer()));
                    }
                    else
                    {
                        auto pctx = gctx.get_profiling_context(outnode);
                        result[outnode->get_unique_name()] =
                            pctx->kernel_memory->save_outputs(type_out);
                    }
                }

                // The result data ptr is like result["nodename"]->kernel_memory->unsafe_output(0);
                return move(result);
            }

            unordered_map<string, vector<vector<char>>>
                mixed_type_eval(const vector<vector<char>>& inputs)
            {
                auto parameters = gctx.graph->get_parameters();

                NNFUSION_CHECK(inputs.size() == parameters.size())
                    << "The input size does not match graph's Parameter count";
                for (size_t i = 0; i < parameters.size(); i++)
                {
                    parameter_map[parameters[i]] = i;
                }
                auto ordered_ops = gctx.graph->get_ordered_ops();
                for (auto& op : ordered_ops)
                {
                    create_profiling_contexts(op);
                }

                for (auto& op : ordered_ops)
                {
                    connect_nodes(op, inputs);
                }

                int i = 0;
                for (auto& node : ordered_ops)
                {
                    if (node->get_op_ptr()->is_tensor_op())
                    {
                        continue;
                    }
                    auto pctx = gctx.get_profiling_context(node);
                    // Ensure only run once
                    pctx->warmup_times = 0;
                    pctx->runtime_times = 1;

                    rt->execute(pctx,
                                pctx->kernel_memory->unsafe_inputs(),
                                pctx->kernel_memory->unsafe_outputs());
                }

                unordered_map<string, vector<vector<char>>> result;
                vector<vector<char>> outputs;
                for (auto& outnode : gctx.graph->get_outputs())
                {
                    outputs.clear();
                    if (outnode->is_parameter())
                    {
                        size_t _size = nnfusion::shape_size(outnode->get_shape()) *
                                       outnode->get_element_type().size();
                        vector<char> output(_size);
                        memcpy(output.data(), (void*)inputs[parameter_map[outnode]].data(), _size);
                        outputs.push_back(move(output));
                    }
                    ///\todo tensor op?
                    else if (outnode->is_constant())
                    {
                        auto const_node = static_pointer_cast<op::Constant>(outnode->get_op_ptr());
                        vector<char> output(const_node->get_data_size());
                        memcpy(
                            output.data(), const_node->get_data_ptr(), const_node->get_data_size());
                        outputs.push_back(move(output));
                    }
                    else
                    {
                        auto pctx = gctx.get_profiling_context(outnode);
                        auto& kernel_mem = pctx->kernel_memory;
                        auto kctx = pctx->kernel->m_context;

                        void** ptrs = kernel_mem->unsafe_outputs();
                        for (size_t i = 0; i < kctx->outputs.size(); ++i)
                        {
                            auto& t = kctx->outputs[i];
                            size_t _size = t->size();

                            NNFUSION_CHECK(ptrs[i] != nullptr);
                            vector<char> output(_size);
                            memcpy(output.data(), ptrs[i], _size);

                            outputs.push_back(move(output));
                        }
                    }
                    result[outnode->get_unique_name()] = outputs;
                }

                // The result data ptr is like result["nodename"]->kernel_memory->unsafe_output(0);
                return move(result);
            }

        private:
            GraphEvaluationContext gctx;
            IProfilingRuntime::Pointer rt;
            NNFusion_DeviceType dev_type;
            std::unordered_map<std::shared_ptr<GNode>, int> parameter_map;

            void create_profiling_contexts(shared_ptr<GNode> node);

            template <typename T>
            void connect_nodes(shared_ptr<GNode> gnode, const vector<vector<T>>& inputs)
            {
                if (gnode->is_parameter())
                {
                    for (auto& edge : gnode->get_out_edges())
                    {
                        // Skip control edge
                        if (edge->is_control_edge())
                            continue;
                        auto dstnode = edge->get_dst();
                        auto dstpctx = gctx.get_profiling_context(dstnode);
                        size_t _size = nnfusion::shape_size(gnode->get_shape()) *
                                       gnode->get_element_type().size();
                        // This statments will remove some allocated memory.
                        dstpctx->kernel_memory->load_input_from(
                            edge->get_dst_input(),
                            (void*)inputs[parameter_map[gnode]].data(),
                            _size);
                    }
                }
                else if (gnode->is_constant())
                {
                    auto const_node = static_pointer_cast<op::Constant>(gnode->get_op_ptr());

                    for (auto& edge : gnode->get_out_edges())
                    {
                        // Skip control edge
                        if (edge->is_control_edge())
                            continue;
                        auto dstnode = edge->get_dst();
                        auto dstpctx = gctx.get_profiling_context(dstnode);
                        // This statments will remove some allocated memory.
                        dstpctx->kernel_memory->load_input_from(edge->get_dst_input(),
                                                                const_node->get_data_ptr(),
                                                                const_node->get_data_size());
                    }
                }
                else
                {
                    auto pctx = gctx.get_profiling_context(gnode);
                    for (auto& edge : gnode->get_out_edges())
                    {
                        // Skip control edge
                        if (edge->is_control_edge())
                            continue;
                        auto dstnode = edge->get_dst();
                        auto dstpctx = gctx.get_profiling_context(dstnode);
                        if (!dstpctx)
                            continue;
                        // This statments will remove some allocated memory.
                        pctx->kernel_memory->forward(
                            edge->get_src_output(), dstpctx->kernel_memory, edge->get_dst_input());
                    }
                }
            }

            void connect_nodes(shared_ptr<GNode> gnode, const vector<DataBuffer>& inputs)
            {
                if (gnode->is_parameter())
                {
                    for (auto& edge : gnode->get_out_edges())
                    {
                        // Skip control edge
                        if (edge->is_control_edge())
                            continue;
                        auto dstnode = edge->get_dst();
                        auto dstpctx = gctx.get_profiling_context(dstnode);
                        size_t _size = nnfusion::shape_size(gnode->get_shape()) *
                                       gnode->get_element_type().size();
                        // This statments will remove some allocated memory.
                        dstpctx->kernel_memory->load_input_from(
                            edge->get_dst_input(), inputs[parameter_map[gnode]].data(), _size);
                    }
                }
                else if (gnode->is_constant())
                {
                    auto const_node = static_pointer_cast<op::Constant>(gnode->get_op_ptr());

                    for (auto& edge : gnode->get_out_edges())
                    {
                        // Skip control edge
                        if (edge->is_control_edge())
                            continue;
                        auto dstnode = edge->get_dst();
                        auto dstpctx = gctx.get_profiling_context(dstnode);
                        // This statments will remove some allocated memory.
                        dstpctx->kernel_memory->load_input_from(edge->get_dst_input(),
                                                                const_node->get_data_ptr(),
                                                                const_node->get_data_size());
                    }
                }
                else
                {
                    auto pctx = gctx.get_profiling_context(gnode);
                    for (auto& edge : gnode->get_out_edges())
                    {
                        // Skip control edge
                        if (edge->is_control_edge())
                            continue;
                        auto dstnode = edge->get_dst();
                        auto dstpctx = gctx.get_profiling_context(dstnode);
                        if (!dstpctx)
                            continue;
                        // This statments will remove some allocated memory.
                        pctx->kernel_memory->forward(
                            edge->get_src_output(), dstpctx->kernel_memory, edge->get_dst_input());
                    }
                }
            }
        };
    };
}
