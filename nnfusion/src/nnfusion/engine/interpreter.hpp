// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/core/IR/program.hpp"
#include "nnfusion/engine/memory_allocator.hpp"
#include "op.hpp"

namespace nnfusion
{
    class InterpreterContext;
    class TranslationUnit;

    class IInterpreterPass
    {
    public:
        virtual bool run(shared_ptr<InterpreterContext> ctx, shared_ptr<TranslationUnit> tu) = 0;
    };

    class TranslationUnit
    {
    public:
        using Pointer = shared_ptr<TranslationUnit>;
        shared_ptr<graph::Graph> m_graph;
        shared_ptr<vector<ir::Operator_p>> inter_ops;
        shared_ptr<set<string>> input_names;
        shared_ptr<set<string>> output_names;
        shared_ptr<set<shared_ptr<nnfusion::descriptor::Tensor>>> constants;
        vector<shared_ptr<nnfusion::descriptor::Tensor>> arg;
        vector<shared_ptr<nnfusion::descriptor::Tensor>> out;
        unordered_set<shared_ptr<graph::GNode>> blacklist;
        shared_ptr<MemoryAllocatorFactory> memory_allocator_factory;
        nnfusion::ir::Program program;
        bool m_is_translated;
        size_t memory_pool_size;
        TranslationUnit()
            : inter_ops(new vector<ir::Operator_p>())
            , memory_pool_size(0)
            , m_is_translated(false)
            , input_names(new set<string>())
            , output_names(new set<string>())
            , constants(new set<shared_ptr<nnfusion::descriptor::Tensor>>()){};
    };

    using TranslationUnitMap = map<shared_ptr<graph::Graph>, shared_ptr<TranslationUnit>>;

    class InterpreterContext
    {
    public:
        //shared_ptr<graph::Graph> m_graph;
        unordered_set<shared_ptr<graph::Graph>> m_graphs;
        // Store Translated OP's
        unordered_map<shared_ptr<graph::GNode>, ir::Operator_p> m_node_inter_map;
        size_t m_offset;
        unordered_map<string, size_t> m_tensor_memory_buffers;
        unordered_map<string, string> m_variable_name_map;
        TranslationUnitMap m_tus;
    };
}