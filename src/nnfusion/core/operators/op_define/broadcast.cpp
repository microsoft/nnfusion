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

#include "broadcast.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

Broadcast::Broadcast(const std::string& name,
                     const nnfusion::Shape& shape,
                     const nnfusion::AxisSet& broadcast_axes)
    : Op(name)
    , m_shape(shape)
    , m_broadcast_axes(broadcast_axes)
{
}

Broadcast::Broadcast(const nnfusion::Shape& shape, const nnfusion::AxisSet& broadcast_axes)
    : Broadcast("Broadcast", shape, broadcast_axes)
{
}

void Broadcast::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    infer_shape(gnode);

    inner_or_outer_broadcast();

    for (auto axis : m_broadcast_axes)
    {
        OP_VALIDATION(this, axis < m_shape.size())
            << "Broadcast axis index (" << axis << ") exceeds specified output shape rank "
            << "(broadcast axes: " << m_broadcast_axes << ", output shape: " << m_shape << ").";
    }

    nnfusion::Shape required_input_shape = m_shape;
    for (auto i = m_broadcast_axes.rbegin(); i != m_broadcast_axes.rend(); ++i)
    {
        required_input_shape.erase(required_input_shape.begin() + *i);
    }

    // TODO(amprocte): We can probably have a more helpful error message here.
    // There are two things that can go wrong, which are being picked up in
    // one fell swoop by this check: either the number of broadcast axes is not
    // enough, or there is a mismatch with one of the pre-broadcast axis lengths.
    auto partial_shape = gnode->get_input_partial_shape(0);
    OP_VALIDATION(this, partial_shape.compatible(required_input_shape))
        << "Broadcast argument shape, specified output shape, and axes are incompatible "
        << "(argument shape: " << partial_shape << ", output shape: " << m_shape
        << ", broadcast axes: " << m_broadcast_axes << ").";

    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), m_shape);
}

void Broadcast::inner_or_outer_broadcast()
{
    nnfusion::AxisSet outer_axes;
    size_t rest_size = 1;
    bool count_size_only = false;
    size_t no_need_broadcast = 0;
    for (size_t i = 0; i < m_shape.size(); i++)
    {
        if ((m_broadcast_axes.count(i) > 0 || m_shape[i] == 1) && !count_size_only)
        {
            outer_axes.insert(i);
            if (m_shape[i] == 1)
                no_need_broadcast++;
        }
        else
        {
            count_size_only = true;
            rest_size *= m_shape[i];
        }
    }
    if (outer_axes.size() == m_broadcast_axes.size() + no_need_broadcast)
    {
        m_is_outer_broadcast = true;
        m_outer_bc_size = rest_size;
        return;
    }

    nnfusion::AxisSet inner_axes;
    for (size_t i = m_shape.size() - 1; i >= 0; i--)
    {
        if (m_broadcast_axes.count(i) > 0)
            inner_axes.insert(i);
        else if (m_shape[i] == 1)
            continue;
        else
            break;
    }
    if (inner_axes.size() == m_broadcast_axes.size())
    {
        m_is_inner_broadcast = true;
        size_t size = 1;
        for (auto d : inner_axes)
            size *= m_shape[d];
        m_inner_bc_size = size;
        return;
    }
}

void Broadcast::infer_shared_memory(std::shared_ptr<graph::GNode> gnode)
{
    m_shared_memory.clear();
    auto& output_shape = gnode->get_output_shape(0);
    for (size_t i = 0; i < output_shape.size(); i++)
    {
        m_shared_memory.push_back(1);
    }
}

BroadcastLike::BroadcastLike(const AxisSet& broadcast_axes)
    : Broadcast("BroadcastLike", {}, {})
    , m_initial_broadcast_axes(broadcast_axes)
{
}

void BroadcastLike::infer_shape(std::shared_ptr<graph::GNode> gnode)
{
    const Shape& in_shape = gnode->get_input_shape(0);
    m_shape = gnode->get_input_shape(1);
    m_broadcast_axes = m_initial_broadcast_axes;
    if (m_broadcast_axes.size() == 0)
    {
        for (size_t i = 0; i < m_shape.size(); ++i)
        {
            if (i < in_shape.size())
            {
                if (in_shape.at(i) == 1 && m_shape.at(i) > 1)
                {
                    m_broadcast_axes.insert(i);
                }
            }
            else
            {
                m_broadcast_axes.insert(i);
            }
        }
    }
}
