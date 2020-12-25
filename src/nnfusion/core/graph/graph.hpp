// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <fstream>
#include <memory>
#include <vector>
#include "gedge.hpp"
#include "gnode.hpp"

namespace nnfusion
{
    namespace graph
    {
        // Thread compatible but not thread safe.
        class Graph
        {
        public:
            using Pointer = std::shared_ptr<Graph>;
            // Constructs a graph with a single SOURCE (always id kSourceId) and a
            // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.

            Graph(const std::string& name = "");

            ~Graph();

            static const int kControlSlot = -1;
            static const int freeGnodeId = -1;

            const std::string& get_friendly_name() const;
            const std::string& get_name() const;
            void set_name(const std::string& name);

            // Adds a new node to this graph, and returns it. Infers the Op and
            // input/output types for the node. *this owns the returned instance.
            // Returns nullptr and sets *status on error.
            void add_node(std::shared_ptr<GNode> node);

            std::shared_ptr<GNode> add_node_and_edge(const std::shared_ptr<nnfusion::op::Op> op,
                                                     const GNodeVector& input_gnodes,
                                                     const size_t output_size = 1);

            std::shared_ptr<GNode> add_node_and_edge(const std::shared_ptr<nnfusion::op::Op> op,
                                                     const GNodeIndexVector& input_gnodes,
                                                     const size_t output_size = 1);

            void add_gnode_and_edge(const std::shared_ptr<GNode> gnode,
                                    const GNodeIndexVector& input_gnodes);

            // Removes a node from this graph, including all edges from or to it.
            // *node should not be accessed after calling this function.
            // REQUIRES: node->IsOp()
            void remove_node(std::shared_ptr<GNode> node);

            void replace_node(std::shared_ptr<GNode> old_node,
                              std::shared_ptr<GNode> new_node,
                              bool copy_in_edge = true);

            // The number of live nodes in the graph.
            //
            // Because nodes can be removed from the graph, get_node_size() is often
            // smaller than get_max_node_id(). If one needs to create an array of
            // nodes indexed by node ids, get_max_node_id() should be used as the
            // array's size.
            size_t get_node_size() const { return m_node_size; }
            // Returns one more than the maximum id assigned to any node.
            size_t get_max_node_id() const { return m_nodes.size(); }
            // Returns the node associated with an id, or nullptr if no node
            // with that id (the node with that id was removed and the id has
            // not yet been re-used). *this owns the returned instance.
            // REQUIRES: 0 <= id < get_max_node_id().

            GNodeVector get_nodes();
            GNodeVector get_ordered_ops(bool include_control_deps = true);
            GNodeVector get_bfs_ordered_ops();

            GNodeVector get_const_nodes();

            std::shared_ptr<GNode> find_node_id(size_t id) const { return m_nodes[id]; }
            // Adds an edge that connects the xth output of `source` to the yth input of
            // `dest` and returns it. Does not update dest's NodeDef.
            const std::shared_ptr<Edge>
                add_edge(std::shared_ptr<GNode> source, int x, std::shared_ptr<GNode> dest, int y);

            bool
                find_edge(std::shared_ptr<GNode> source, int x, std::shared_ptr<GNode> dest, int y);
            const std::shared_ptr<Edge> add_control_edge(std::shared_ptr<GNode> source,
                                                         std::shared_ptr<GNode> dest,
                                                         bool allow_duplicates = false);
            // Removes edge from the graph. Does not update the destination node's
            // NodeDef.
            // REQUIRES: The edge must exist.
            void remove_edge(const std::shared_ptr<Edge> edge);
            // Updates the input to a node.  The existing edge to `dst` is removed and an
            // edge from `new_src` to `dst` is created. The NodeDef associated with `dst`
            // is also updated.
            //Status UpdateEdge(std::shared_ptr<Node> new_src, int new_src_index, std::shared_ptr<Node> dst, int dst_index);

            // The number of live edges in the graph.
            //
            // Because edges can be removed from the graph, get_edge_size() is often
            // smaller than get_max_edge_id(). If one needs to create an array of
            // edges indexed by edge ids, get_max_edge_id() should be used as the
            // array's size.
            size_t get_edge_size() const { return m_edge_size; }
            // Returns one more than the maximum id assigned to any edge.
            size_t get_max_edge_id() const { return m_edges.size(); }
            // Returns the Edge associated with an id, or nullptr if no edge
            // with that id (the node with that id was removed and the id has
            // not yet been re-used). *this owns the returned instance.
            // REQUIRES: 0 <= id < get_max_node_id().
            const std::shared_ptr<Edge> find_edge_id(size_t id) const { return m_edges[id]; }
            GNodeVector get_outputs();

            void set_outputs(const GNodeVector& outputs);
            void set_default_outputs();
            const size_t get_output_size();
            /// Return the op that generates output i
            const std::shared_ptr<GNode> get_output_op(size_t i);

            GNodeVector get_parameters();
            void set_default_parameters();

            size_t get_temporary_pool_size();
            void set_temporary_pool_size(size_t);

            size_t get_memory_io();
            bool serialize_to_file(const std::string& file_path);

        private:
            // Map from node ids to allocated nodes.  nodes_[id] may be nullptr if
            // the node with that id was removed from the graph.
            GNodeVector m_nodes;

            //ordered ops from bfs
            GNodeVector m_bfs_ordered_ops;
            bool m_bfs_ordered_ops_is_valid = false;

            // Number of nodes alive.
            size_t m_node_size = 0;

            // Map from edge ids to allocated edges.  m_edges[id] may be nullptr if
            // the edge with that id was removed from the graph.
            std::vector<std::shared_ptr<Edge>> m_edges;

            // The number of entries in m_edges that are not nullptr.
            size_t m_edge_size = 0;

            // Allocated but free nodes and edges.
            GNodeVector m_free_nodes;
            std::vector<std::shared_ptr<Edge>> m_free_edges;

            // TODO: Output nodes of this graph
            GNodeVector m_output_nodes;
            GNodeVector m_parameters;
            // For generating unique names.
            int name_counter_ = 0;

            static std::atomic<size_t> m_next_instance_id;
            size_t m_instance_id;
            std::string m_name;
            const std::string m_unique_name;

            size_t m_temporary_pool_size;
        };

        inline bool Edge::is_control_edge() const
        {
            // Note that if either src_output_ or dst_input_ is kControlSlot,
            // so is the other one (add_edge checks this).
            return m_src_output == Graph::kControlSlot;
        }
    }
}
