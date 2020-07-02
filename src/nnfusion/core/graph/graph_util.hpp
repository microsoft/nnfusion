#pragma once

#include "gedge.hpp"
#include "gnode.hpp"
#include "graph.hpp"

namespace nnfusion
{
    namespace graph
    {
        // Comparator for two nodes. This is used in order to get a stable ording.
        using NodeComparator =
            std::function<bool(const std::shared_ptr<GNode>, const std::shared_ptr<GNode>)>;

        // Compares two node based on their ids.
        struct NodeComparatorID
        {
            bool operator()(const std::shared_ptr<GNode> n1, const std::shared_ptr<GNode> n2) const
            {
                return n1->get_id() < n2->get_id();
            }
        };

        // Compare two nodes based on their names.
        struct NodeComparatorName
        {
            bool operator()(const std::shared_ptr<GNode> n1, const std::shared_ptr<GNode> n2) const
            {
                return n1->get_name() < n2->get_name();
            }
        };

        void ReverseDFS(const Graph* graph,
                        const GNodeVector& start,
                        const std::function<void(std::shared_ptr<GNode>)>& enter,
                        const std::function<void(std::shared_ptr<GNode>)>& leave,
                        const NodeComparator& stable_comparator);
    }
}
