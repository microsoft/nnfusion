// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion;

class BertFusionOptimizer
{
public:
    BertFusionOptimizer(std::shared_ptr<nnfusion::graph::Graph> g)
        : m_graph(g)
    {
    }

    virtual bool Optimize() { return false; }
protected:
    std::shared_ptr<nnfusion::graph::Graph> m_graph;
};