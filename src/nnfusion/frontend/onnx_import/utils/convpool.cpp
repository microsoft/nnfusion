// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cmath>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/shape.hpp"

#include "convpool.hpp"
#include "core/attribute.hpp"
#include "core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace convpool
        {
            Shape get_kernel_shape(const Node& node)
            {
                std::size_t input_spacial_dims = node.get_ng_inputs()[0]->get_shape().size() - 2;
                return node.get_attribute_value<std::vector<std::size_t>>(
                    "kernel_shape", std::vector<std::size_t>(input_spacial_dims, 1UL));
            }

            namespace detail
            {
                Strides get_strides_helper(const Node& node,
                                           const std::string& name,
                                           const Shape& kernel_shape)
                {
                    return node.get_attribute_value<std::vector<std::size_t>>(
                        name, std::vector<std::size_t>(kernel_shape.size(), 1UL));
                }
            } // namespace detail

            Strides get_strides(const Node& node, const Shape& kernel_shape)
            {
                return detail::get_strides_helper(node, "strides", kernel_shape);
            }

            Strides get_strides(const Node& node)
            {
                return get_strides(node, get_kernel_shape(node));
            }

            Strides get_dilations(const Node& node)
            {
                return detail::get_strides_helper(node, "dilations", get_kernel_shape(node));
            }

            namespace
            {
                CoordinateDiff get_auto_pads(const Shape& kernel_shape, const std::string& auto_pad)
                {
                    CoordinateDiff pads;

                    // Add padding to the input to match the size of output size.
                    auto pad_value = [](size_t dim) {
                        return (static_cast<float>(dim) - 1.f) / 2.f;
                    };

                    if (auto_pad == "SAME_UPPER")
                    {
                        for (size_t dim : kernel_shape)
                        {
                            pads.emplace_back(std::floor(pad_value(dim)));
                        }
                        for (size_t dim : kernel_shape)
                        {
                            pads.emplace_back(std::ceil(pad_value(dim)));
                        }
                    }
                    else if (auto_pad == "SAME_LOWER")
                    {
                        for (size_t dim : kernel_shape)
                        {
                            pads.emplace_back(std::ceil(pad_value(dim)));
                        }
                        for (size_t dim : kernel_shape)
                        {
                            pads.emplace_back(std::floor(pad_value(dim)));
                        }
                    }

                    return pads;
                }

            } // namespace

            std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node,
                                                               const Shape& kernel_shape)
            {
                CoordinateDiff pads;
                try
                {
                    auto pads_int64 = node.get_attribute_value<std::vector<int64_t>>("pads");
                    pads = CoordinateDiff{std::begin(pads_int64), std::end(pads_int64)};
                }
                catch (const error::node::UnknownAttribute&)
                {
                    std::string auto_pad{node.get_attribute_value<std::string>("auto_pad", "")};
                    if (!auto_pad.empty())
                    {
                        pads = get_auto_pads(kernel_shape, auto_pad);
                    }
                }
                if (pads.empty())
                {
                    pads = CoordinateDiff(static_cast<std::ptrdiff_t>(kernel_shape.size()), 0UL);
                }

                if (pads.size() != kernel_shape.size() * 2)
                {
                    // Paddings specified in (H, W, C) format.
                    return {pads, pads};
                }
                else
                {
                    return {{std::begin(pads) + pads.size() / 2, std::end(pads)},
                            {std::begin(pads), std::begin(pads) + pads.size() / 2}};
                }
            }

        } // namespace convpool
    }     // namespace onnx_import
} // namespace ngraph
