//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsum.hpp"
#include <cctype>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_12
            {
                namespace
                {
                    auto get_between = [](const std::string& str,
                                          const std::string& begin,
                                          const std::string& end,
                                          int start_idx = 0,
                                          const std::string& def_ret = "") -> std::string {
                        if (start_idx < 0)
                            return def_ret;
                        int at = str.find(begin);
                        if (at < 0)
                            return def_ret;
                        at += begin.size();
                        int next = str.find(end, at);
                        if (next < at)
                            return def_ret;
                        return str.substr(at, next - at);
                    };

                    auto ssplit = [](const std::string& str,
                                     const std::string& sub) -> std::vector<std::string> {
                        std::vector<std::string> ret;
                        int it = 0, next;
                        while (next = str.find(sub, it), next >= 0)
                        {
                            ret.push_back(str.substr(it, next - it));
                            it = next + sub.size();
                        }
                        ret.push_back(str.substr(it));
                        return std::move(ret);
                    };

                    auto sstrip = [](const std::string& str) -> std::string {
                        auto start_it = str.begin();
                        auto end_it = str.rbegin();
                        while (std::isspace(*start_it))
                            ++start_it;
                        while (std::isspace(*end_it))
                            ++end_it;
                        return std::string(start_it, end_it.base());
                    };
                } // namespace

                NamedNodeVector TranslateEinsumOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    Node node(node_proto);
                    auto equation = node.get_attribute_value<std::string>("equation");

                    std::vector<std::vector<std::string>> input_layout;
                    std::vector<std::string> output_layout;
                    std::vector<std::string> ellipsis_layout;

                    if (equation.find("->") != string::npos)
                    {
                        // explicit mode like "...ii->...i"
                        auto sub_eq = ssplit(equation, "->");
                        auto input_expr = ssplit(sub_eq[0], ",");
                        for (size_t i = 0; i < input_expr.size(); i++)
                        {
                            auto input_i = sstrip(input_expr[i]);
                            std::vector<std::string> input_i_layout;
                            if (input_i.find("...") != string::npos)
                            {
                                // handle abc...def
                                auto left_right = ssplit(input_i, "...");
                                auto left = left_right[0];
                                auto right = left_right[1];
                                for (auto c : left)
                                {
                                    input_i_layout.emplace_back(1, toupper(c));
                                }

                                bool flag = ellipsis_layout.empty();
                                for (size_t ii = 0; ii < input_indexes[i].get_shape().size() -
                                                             left.size() - right.size();
                                     ii++)
                                {
                                    input_i_layout.push_back("N" + std::to_string(ii));
                                    if (flag)
                                    {
                                        ellipsis_layout.push_back("N" + std::to_string(ii));
                                    }
                                }
                                NNFUSION_CHECK(ellipsis_layout.size() ==
                                               input_indexes[i].get_shape().size() - left.size() -
                                                   right.size())
                                    << "ellipsis in einsum equation should represent of the same "
                                       "rank";

                                for (auto c : right)
                                {
                                    input_i_layout.emplace_back(1, toupper(c));
                                }
                            }
                            else
                            {
                                for (auto c : input_i)
                                {
                                    input_i_layout.emplace_back(1, toupper(c));
                                }
                            }
                            input_layout.push_back(input_i_layout);
                        }

                        auto output_expr = sstrip(sub_eq[1]);
                        if (output_expr.find("...") != string::npos)
                        {
                            // handle abc...def
                            auto left_right = ssplit(output_expr, "...");
                            auto left = left_right[0];
                            auto right = left_right[1];
                            for (auto c : left)
                            {
                                output_layout.emplace_back(1, toupper(c));
                            }

                            for (auto index : ellipsis_layout)
                            {
                                output_layout.push_back(index);
                            }

                            for (auto c : right)
                            {
                                output_layout.emplace_back(1, toupper(c));
                            }
                        }
                        else
                        {
                            for (auto c : output_expr)
                            {
                                output_layout.emplace_back(1, toupper(c));
                            }
                        }
                    }
                    else
                    {
                        // implicit mode like "...i,...i"
                        std::unordered_map<std::string, size_t> indices_cnt;
                        auto input_expr = ssplit(equation, ",");
                        for (size_t i = 0; i < input_expr.size(); i++)
                        {
                            auto input_i = sstrip(input_expr[i]);
                            std::vector<std::string> input_i_layout;
                            if (input_i.find("...") != string::npos)
                            {
                                // handle abc...def
                                auto left_right = ssplit(input_i, "...");
                                auto left = left_right[0];
                                auto right = left_right[1];
                                for (auto c : left)
                                {
                                    input_i_layout.emplace_back(1, toupper(c));
                                    indices_cnt[std::string(1, toupper(c))] += 1;
                                }

                                bool flag = ellipsis_layout.empty();
                                for (size_t ii = 0; ii < input_indexes[i].get_shape().size() -
                                                             left.size() - right.size();
                                     ii++)
                                {
                                    input_i_layout.push_back("N" + std::to_string(ii));
                                    if (flag)
                                    {
                                        ellipsis_layout.push_back("N" + std::to_string(ii));
                                    }
                                }
                                NNFUSION_CHECK(ellipsis_layout.size() ==
                                               input_indexes[i].get_shape().size() - left.size() -
                                                   right.size())
                                    << "ellipsis in einsum equation should represent of the same "
                                       "rank";

                                for (auto c : right)
                                {
                                    input_i_layout.emplace_back(1, toupper(c));
                                    indices_cnt[std::string(1, toupper(c))] += 1;
                                }
                            }
                            else
                            {
                                for (auto c : input_i)
                                {
                                    input_i_layout.emplace_back(1, toupper(c));
                                    indices_cnt[std::string(1, toupper(c))] += 1;
                                }
                            }
                            input_layout.push_back(input_i_layout);
                        }

                        // deduce output layout
                        // output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the equation
                        // an index occuring multiple times in input_expr is a reduction axis and won't appear in output
                        std::vector<std::string> keeping_indices;
                        for (auto const& x : indices_cnt)
                        {
                            if (x.second == 1)
                            {
                                keeping_indices.push_back(x.first);
                            }
                        }
                        std::sort(keeping_indices.begin(), keeping_indices.end());
                        // In implicit mode, the ellipsis dimensions are set to the beginning of the output
                        for (auto index : ellipsis_layout)
                        {
                            output_layout.push_back(index);
                        }
                        for (auto index : keeping_indices)
                        {
                            output_layout.push_back(index);
                        }
                    }

                    // // print layout
                    // for (size_t i=0; i<input_layout.size(); i++)
                    // {
                    //     cout << "layout of input_" << i << ": [";
                    //     for (size_t ii=0; ii<input_layout[i].size(); ii++)
                    //     {
                    //         cout << input_layout[i][ii];
                    //         if (ii + 1 < input_layout[i].size())
                    //         {
                    //             cout << ", ";
                    //         }
                    //     }
                    //     cout << "]" << endl;
                    // }

                    // cout << "layout of output" << ": [";
                    // for (size_t ii=0; ii<output_layout.size(); ii++)
                    // {
                    //     cout << output_layout[ii];
                    //     if (ii + 1 < output_layout.size())
                    //     {
                    //         cout << ", ";
                    //     }
                    // }
                    // cout << "]" << endl;

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["input_layout"] = input_layout;
                    myConfig["output_layout"] = output_layout;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "Einsum", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes);

                    return {{node_proto.output(0), generic_gnode}};
                }

            } // namespace set_12

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
