// Microsoft (c) 2019, NNFusion Team

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

#include "util/annotations.hpp"

#include "nnfusion/util/util.hpp"

#include "nnfusion/common/partial_shape.hpp"
#include "nnfusion/common/type/element_type.hpp"

namespace nnfusion
{
    namespace graph
    {
        class GNode;
    }

    namespace op
    {
        class Op;
    }

    std::string op_validation_string(const op::Op* op);

    namespace op
    {
        class Op : public std::enable_shared_from_this<Op>
        {
        public:
            virtual ~Op();
            void revalidate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
            {
                validate_and_infer_types(gnode);
            }
            // Called after transition
            void delayed_validate_and_infer_types(std::shared_ptr<graph::GNode> gnode);

            /// The class name, must not contain spaces
            std::string get_op_type() const { return m_op_type; }
            const std::string& get_name() const;
            const std::string& get_unique_name() const;
            void set_name(const std::string& name);
            /// Return true if this has the same implementing class as op. This
            /// will be used by the pattern matcher when comparing a pattern
            /// graph against the graph.
            bool is_same_op_type(const std::shared_ptr<Op>& op) const
            {
                Op* op_ptr = op.get();
                return std::type_index(typeid(*this)) == std::type_index(typeid(*op_ptr));
            }

            virtual bool is_parameter() const;
            virtual bool is_output() const;
            virtual bool is_constant() const;
            virtual bool is_commutative() { return false; }
            size_t get_instance_id() const { return m_instance_id; }
            size_t get_id() const { return m_id; }
            size_t set_id(size_t id)
            {
                m_id = id;
                return m_id;
            }

            void Clear();

            virtual std::shared_ptr<Op> get_default_value() const { return nullptr; }
            /// Use instance ids for comparison instead of memory addresses to improve determinism
            bool operator<(const Op& other) const { return m_instance_id < other.m_instance_id; }
            void set_op_annotations(std::shared_ptr<Annotations> op_annotations)
            {
                m_op_annotations = op_annotations;
            }
            std::shared_ptr<Annotations> get_op_annotations() const { return m_op_annotations; }
        protected:
            Op(const std::string& op_type);

            virtual void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode);

            // Called in constructors during transition
            void constructor_validate_and_infer_types(std::shared_ptr<graph::GNode> gnode);

            std::tuple<nnfusion::element::Type, nnfusion::PartialShape>
                validate_and_infer_elementwise_args(std::shared_ptr<graph::GNode> gnode);

            std::string m_op_type;
            size_t m_instance_id;
            size_t m_id; // m_id is for graph, the index in graph m_nodes
            std::string m_name;
            const std::string m_unique_name;
            static std::atomic<size_t> m_next_instance_id;

        private:
            std::shared_ptr<Annotations> m_op_annotations;
        };
    }
}

#define OP_VALIDATION(op, cond)                                                                    \
    NNFUSION_CHECK_STREAM_WITH_LOC(                                                                \
        ::nnfusion::errors::CheckError, cond, ::nnfusion::op_validation_string(op))
