// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gnode_vector.hpp"
#include "nnfusion/core/graph/input.hpp"
#include "nnfusion/core/graph/output.hpp"

#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/core/IR/attribute.hpp"
#include "nnfusion/core/operators/op.hpp"
namespace nnfusion
{
    namespace graph
    {
        class Edge;
        class GNode;
        class Graph;
        struct GNodeIndex
        {
            explicit GNodeIndex(std::shared_ptr<GNode> gnode, int i)
                : gnode(gnode)
                , index(i)
            {
            }
            ///\todo remove explicit
            explicit GNodeIndex(std::shared_ptr<GNode> gnode)
                : GNodeIndex(gnode, 0)
            {
            }
            GNodeIndex()
                : GNodeIndex(nullptr)
            {
            }
            GNodeIndex& operator=(const GNodeIndex& gnode_index) = default;
            bool operator==(const GNodeIndex& other) const
            {
                return gnode == other.gnode && index == other.index;
            }
            bool operator!=(const GNodeIndex& other) const { return !(*this == other); }
            bool empty() const { return gnode == nullptr; }
            const nnfusion::Shape& get_shape() const;
            const nnfusion::element::Type& get_element_type() const;
            std::shared_ptr<nnfusion::graph::GNode> gnode;
            int index;
        };
        using GNodeIndexVector = std::vector<GNodeIndex>;
        /// Nodes are the backbone of the graph of Value dataflow. Every node has
        /// zero or more nodes as arguments and one value, which is either a tensor
        /// view or a (possibly empty) tuple of values.
        class GNode : public std::enable_shared_from_this<GNode>, public nnfusion::ir::Tagable
        {
        public:
            GNode();
            GNode(const std::shared_ptr<op::Op> op_ptr,
                  const GNodeVector& input_gnodes,
                  size_t output_size = 1);
            GNode(const std::shared_ptr<op::Op> op_ptr,
                  const GNodeIndexVector& input_gnode_indexs,
                  size_t output_size = 1);

            ~GNode();
            void construct_from_op_ptr(const std::shared_ptr<op::Op>& op_ptr);
            void initialize(const std::shared_ptr<op::Op> op_ptr,
                            const GNodeVector& input_gnodes,
                            size_t output_size = 1);
            void initialize(const std::shared_ptr<op::Op> op_ptr,
                            const GNodeIndexVector& input_gnode_indexs,
                            size_t output_size = 1);
            size_t get_instance_id() const { return m_instance_id; }
            int64_t get_id() const { return m_id; }
            int64_t set_id(int64_t id);

            /// The class name, must not contain spaces
            std::string get_op_type() const { return m_op_type; }
            std::shared_ptr<op::Op> get_op_ptr() const { return m_op_ptr; }
            const std::string& get_unique_name() const;
            const std::string& get_name() const;
            const std::string& get_member_name() const;

            void set_name(const std::string& name);
            void set_member_name(const std::string& name);
            bool has_same_type(std::shared_ptr<const GNode> gnode) const;

            /// in edges
            std::vector<std::shared_ptr<nnfusion::graph::Edge>> get_in_edges() const;
            const std::shared_ptr<nnfusion::graph::Edge> get_in_edge(size_t i) const;

            void add_in_edge(std::shared_ptr<nnfusion::graph::Edge> edge);
            void remove_in_edge(std::shared_ptr<nnfusion::graph::Edge> edge);

            ///  out edges
            std::vector<std::shared_ptr<nnfusion::graph::Edge>> get_out_edges() const;
            std::vector<std::shared_ptr<nnfusion::graph::Edge>>
                get_output_users(size_t i, bool include_control_edge = true);

            void add_out_edge(std::shared_ptr<nnfusion::graph::Edge> edge);
            void remove_out_edge(std::shared_ptr<nnfusion::graph::Edge> edge);

            /// inputs
            std::vector<std::shared_ptr<Input>>& get_inputs() { return m_inputs; }
            const std::vector<std::shared_ptr<Input>>& get_inputs() const { return m_inputs; }
            size_t get_input_size() const { return m_inputs.size(); }
            void set_input_size(size_t n);
            /// Returns the tensor for input i
            nnfusion::descriptor::Tensor& get_input_tensor(size_t i) const;
            /// Returns the tensor view of input i
            std::shared_ptr<nnfusion::descriptor::Tensor> get_input_tensor_ptr(size_t i) const;

            void set_input(size_t i, std::shared_ptr<Input> input);

            /// outputs
            std::vector<std::shared_ptr<Output>>& get_outputs() { return m_outputs; }
            const std::vector<std::shared_ptr<Output>>& get_outputs() const { return m_outputs; }
            size_t get_output_size() const { return m_outputs.size(); }
            void set_output_size(size_t n);
            /// Returns the tensor for output i
            nnfusion::descriptor::Tensor& get_output_tensor(size_t i) const;
            /// Returns the tensor view of output i
            std::shared_ptr<nnfusion::descriptor::Tensor> get_output_tensor_ptr(size_t i) const;

            void set_output(size_t i, std::shared_ptr<Output> output);
            void set_output_type_and_shape(size_t i,
                                           const nnfusion::element::Type& element_type,
                                           const nnfusion::PartialShape& pshape);
            /// Checks that there is exactly one output and returns its shape
            const nnfusion::Shape& get_shape() const;

            /// Checks that there is exactly one output and returns its element type
            const nnfusion::element::Type& get_element_type() const;

            /// Returns the element type for output i
            const nnfusion::element::Type& get_output_element_type(size_t i) const;

            /// Returns the shape for output i
            const nnfusion::Shape& get_output_shape(size_t i) const;

            /// Returns the partial shape for output i
            const nnfusion::PartialShape& get_output_partial_shape(size_t i) const;

            /// Returns the element type of input i
            const nnfusion::element::Type& get_input_element_type(size_t i) const;

            /// Returns the shape of input i
            const nnfusion::Shape& get_input_shape(size_t i) const;

            /// Returns the partial shape of input i
            const nnfusion::PartialShape& get_input_partial_shape(size_t i) const;

            void Clear();

            bool is_constant() const { return m_op_ptr->is_constant(); }
            bool is_variable() const { return m_op_ptr->is_variable(); }
            bool is_parameter() const { return m_op_ptr->is_parameter(); }
            /// Use instance ids for comparison instead of memory addresses to improve determinism
            bool operator<(const GNode& other) const { return m_instance_id < other.m_instance_id; }
            // // \todo make the hash code of different shared_ptr that point to the same address the same
            // std::unordered_set<std::shared_ptr<descriptor::Tensor>> liveness_new_list;
            // std::unordered_set<std::shared_ptr<descriptor::Tensor>> liveness_free_list;

            void set_implementation(std::string impl) { m_implementation = impl; };
            std::string get_implementation() { return m_implementation; };
            void add_symbol(SymDim symbol) { m_symbols.insert(symbol); }
            std::set<SymDim>& get_symbols() { return m_symbols; }
            static void reset_next_instance_id();

        protected:
            int64_t m_id; // m_id is for graph, the index in graph m_nodes
            size_t m_instance_id;
            static std::atomic<size_t> m_next_instance_id;
            std::string m_name;
            std::string m_member_name;
            const std::string m_unique_name;
            std::string m_implementation;

            std::string m_op_type;
            std::shared_ptr<op::Op> m_op_ptr;

            std::set<std::shared_ptr<Edge>> m_in_edges;
            std::set<std::shared_ptr<Edge>> m_out_edges;

            std::vector<std::shared_ptr<Input>> m_inputs;
            std::vector<std::shared_ptr<Output>> m_outputs;

            std::set<SymDim> m_symbols;
        };

        class OpContext
        {
        public:
            OpContext(std::shared_ptr<GNode> gnode)
                : op(gnode->get_op_ptr())
            {
                // extract input tensors
                for (size_t i = 0; i < gnode->get_input_size(); ++i)
                {
                    std::shared_ptr<descriptor::Tensor> tv = gnode->get_input_tensor_ptr(i);
                    NNFUSION_CHECK_NOT_NULLPTR(tv);
                    inputs.push_back(tv);
                }

                // extract output tensors
                for (size_t i = 0; i < gnode->get_output_size(); ++i)
                {
                    std::shared_ptr<descriptor::Tensor> tv = gnode->get_output_tensor_ptr(i);
                    NNFUSION_CHECK_NOT_NULLPTR(tv);
                    outputs.push_back(tv);
                }
            }

            std::shared_ptr<op::Op> op;
            std::vector<std::shared_ptr<nnfusion::descriptor::Tensor>> inputs;
            std::vector<std::shared_ptr<nnfusion::descriptor::Tensor>> outputs;
        };

        class FusedGNode : public GNode
        {
        public:
            FusedGNode(const std::shared_ptr<op::Op> op_ptr)
                : GNode()
            {
                construct_from_op_ptr(op_ptr);
            };

            void build_fused_node(std::unordered_set<std::shared_ptr<GNode>> nodes,
                                  std::shared_ptr<Graph> graph,
                                  bool clean_graph = true);
            std::vector<std::shared_ptr<OpContext>>& get_op_contexts() { return m_op_ctxs; }
        protected:
            void reorder_nodes(std::unordered_set<std::shared_ptr<GNode>> nodes,
                               std::shared_ptr<Graph> graph);
            void set_inputs_and_outputs(std::shared_ptr<Graph> graph);
            void derive_op_def();
            void clean_nodes(std::shared_ptr<Graph> graph);

            std::vector<std::shared_ptr<GNode>> m_order_nodes;
            std::vector<std::shared_ptr<OpContext>> m_op_ctxs;
        };
    } // namespace graph
} // namespace nnfusion
