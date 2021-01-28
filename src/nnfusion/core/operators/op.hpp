//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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

#include "nlohmann/json.hpp"
#include "nnfusion/util/util.hpp"

#include "nnfusion/common/partial_shape.hpp"
#include "nnfusion/common/type/element_type.hpp"

namespace nnfusion
{
    using json = nlohmann::json;

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

            // Used for: 1) Infershape/Translation; 2) Kernel DB Attrs (e.g. get_op_ptr()->serialize().dump()); 3) Checkpoint to file;
            virtual nnfusion::json serialize()
            {
                throw std::runtime_error("Serialize method is not defined for operator type: " +
                                         m_op_type);
            }

            // Used for: 1) Build/Restore node properties from file;
            virtual void deserialize(const nnfusion::json& stat)
            {
                throw std::runtime_error("Deserialize method is not defined for operator type: " +
                                         m_op_type);
            }

            virtual bool is_parameter() const;
            virtual bool is_output() const;
            virtual bool is_constant() const;
            virtual bool is_variable() const;
            virtual bool is_tensor_op() const;
            virtual bool is_commutative();
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
            virtual void infer_shared_memory(std::shared_ptr<graph::GNode> gnode);
            std::vector<size_t> get_shared_memory() const { return m_shared_memory; }
            void set_shared_memory(std::vector<size_t> shared_memory)
            {
                m_shared_memory = shared_memory;
            }

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
            std::vector<size_t> m_shared_memory; // for reduce fusion

        private:
            std::shared_ptr<Annotations> m_op_annotations;
        };
    }
}

#define OP_VALIDATION(op, cond)                                                                    \
    _CHECK_STREAM_WITH_LOC(                                                                        \
        ::nnfusion::errors::CheckError, cond, ::nnfusion::op_validation_string(op))
