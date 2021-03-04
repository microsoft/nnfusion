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

#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "nnfusion/common/device_type.hpp"
#include "nnfusion/common/partial_shape.hpp"
#include "nnfusion/common/shape.hpp"
#include "nnfusion/common/type/element_type.hpp"

namespace nnfusion
{
    //class Node;

    namespace descriptor
    {
        namespace layout
        {
            class TensorLayout;
        }

        /// \brief Compile-time descriptor of a first-class value that is a view of a tensor.
        class Tensor
        {
            Tensor(const Tensor&) = delete;
            Tensor& operator=(const Tensor&) = delete;

        public:
            Tensor(const nnfusion::element::Type& element_type,
                   const nnfusion::PartialShape& pshape,
                   const std::string& name,
                   NNFusion_DeviceType device_type = UNKNOWN,
                   bool is_persistent = false,
                   bool is_constant = false,
                   bool is_parameter = false,
                   bool is_RDMA_tensor = false,
                   const std::string& group = "",
                   int device_id = -1);

            const std::string& get_name(bool get_valid_name = true) const;
            void set_name(const std::string& name) { m_name = name; }
            void set_tensor_type(const nnfusion::element::Type& element_type,
                                 const nnfusion::PartialShape& pshape);

            const nnfusion::element::Type& get_element_type() const { return m_element_type; }
            const nnfusion::Shape& get_shape() const;
            const nnfusion::PartialShape& get_partial_shape() const { return m_partial_shape; }
            const std::shared_ptr<layout::TensorLayout>& get_tensor_layout() const
            {
                return m_tensor_layout;
            }

            void set_tensor_layout(const std::shared_ptr<layout::TensorLayout>& tensor_layout);

            void set_pool_offset(size_t);
            size_t get_pool_offset() const;
            void set_pool(const std::string& pool);
            const std::string& get_pool() const;
            bool is_same_address(std::shared_ptr<Tensor> tensor);
            size_t size(bool in_byte = true) const;

            void set_root_tensor(std::shared_ptr<Tensor> root_tensor)
            {
                m_root_tensor = root_tensor;
            }
            std::shared_ptr<Tensor> get_root_tensor() const { return m_root_tensor; }
            size_t ref() { return ++m_ref_count; }
            size_t deref()
            {
                NNFUSION_CHECK(m_ref_count > 0);
                return --m_ref_count;
            }

            // persistent tensors exist in all iterations, and do not reuse any memory space.
            // Data in persistent tensors can be immutable or mutable.
            bool is_persistent() const { return m_persistent; }
            // Constant tensors contain immutable data.
            bool is_constant() const { return m_constant; }
            bool is_parameter() const { return m_parameter; }
            bool is_RDMA_tensor() const { return m_RDMA; }
            bool is_memset() const { return m_memset; }
            int get_memset_value() const { return m_memset_value; }
            void set_persistent(bool value = true) { m_persistent = value; }
            void set_constant(bool value = true) { m_constant = value; }
            void set_parameter(bool value = true) { m_parameter = value; }
            void set_RDMA(bool value = true) { m_RDMA = value; }
            void set_memset(bool flag = true, int value = 0)
            {
                m_memset = flag;
                m_memset_value = 0;
            }
            void set_group(const std::string& group) { m_group = group; }
            const std::string& get_group() const { return m_group; }
            void set_device_type(NNFusion_DeviceType device_type) { m_device_type = device_type; }
            NNFusion_DeviceType get_device_type() const { return m_device_type; }
            void set_device_id(int device_id) { m_device_id = device_id; }
            int get_device_id() const { return m_device_id; }
            std::string get_device_name() const;
            const std::string& get_unique_name() const { return m_unique_name; }
            using Pointer = std::shared_ptr<Tensor>;

        protected:
            nnfusion::element::Type m_element_type;

            // TODO(amprocte): For now we are maintaining both m_shape and m_partial_shape fields,
            //    with m_shape possibly being invalid (get_shape will throw an exception if it
            //    is). This is because get_shape() returns a const reference. I think ideally we
            //    should refactor so that get_shape returns by value.
            nnfusion::Shape m_shape;
            nnfusion::PartialShape m_partial_shape;

            std::string m_name;
            std::shared_ptr<layout::TensorLayout> m_tensor_layout;
            size_t m_pool_offset{SIZE_MAX};
            std::string m_pool;
            bool m_persistent;
            bool m_constant;
            bool m_parameter;
            bool m_RDMA;
            bool m_memset;
            int m_memset_value;
            std::shared_ptr<Tensor> m_root_tensor;
            size_t m_ref_count;
            std::string m_group;
            NNFusion_DeviceType m_device_type;
            int m_device_id;
            size_t m_instance_id;
            static std::atomic<size_t> m_next_instance_id;
            const std::string m_unique_name;
        };

        std::ostream& operator<<(std::ostream&, const nnfusion::descriptor::Tensor&);
    }
}
