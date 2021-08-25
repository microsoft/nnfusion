// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>
#include "gnode.hpp"

namespace nnfusion
{
    namespace graph
    {
        class Edge
        {
        public:
            std::shared_ptr<GNode> get_src() const { return m_src; }
            std::shared_ptr<GNode> get_dst() const { return m_dst; }
            size_t get_id() const { return m_id; }
            // Return the index of the source output that produces the data
            // carried by this edge.  The special value kControlSlot is used
            // for control dependencies.
            int get_src_output() const { return m_src_output; }
            // Return the index of the destination input that consumes the data
            // carried by this edge.  The special value kControlSlot is used
            // for control dependencies.
            int get_dst_input() const { return m_dst_input; }
            // Return true iff this is an edge that indicates a control-flow
            // (as opposed to a data-flow) dependency.
            bool is_control_edge() const;

            std::string DebugString() const;

        private:
            friend class Graph;
            std::shared_ptr<GNode> m_src;
            std::shared_ptr<GNode> m_dst;
            size_t m_id;
            int m_src_output;
            int m_dst_input;
        };

        struct EdgeComparatorSrcIndex
        {
            bool operator()(const std::shared_ptr<Edge> lhs, const std::shared_ptr<Edge> rhs) const
            {
                return lhs->get_src_output() < rhs->get_src_output();
            }
        };

        struct EdgeComparatorDstIndex
        {
            bool operator()(const std::shared_ptr<Edge> lhs, const std::shared_ptr<Edge> rhs) const
            {
                return lhs->get_dst_input() < rhs->get_dst_input();
            }
        };
    }
}