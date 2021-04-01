// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "inplace_tensor_analysis.hpp"

#include <exception>
#include <queue>
#include <sstream>
#include <utility>

#include "nnfusion/common/util.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

#include "nnfusion/core/operators/op_define/concat.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"

using namespace std;
using namespace nnfusion;
using namespace nnfusion::pass;
using namespace nnfusion::kernels;

bool InplaceTensorAnalysis::run(std::shared_ptr<InterpreterContext> ctx,
                                std::shared_ptr<TranslationUnit> tu)
{
    auto is_same_dev = [](shared_ptr<const descriptor::Tensor> a,
                          shared_ptr<const descriptor::Tensor> b) {
        return (a->get_device_type() == b->get_device_type()) &&
               (a->get_device_id() == b->get_device_id());
    };

    auto& p = tu->program;

    // auto get_kernel = [](std::shared_ptr<ir::Instruction> ins) {
    //     auto emitted_kernel = (*ins)["Kernel_Selection_Result"]
    //                               .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
    //     KernelEmitter::Pointer kernel = nullptr;

    //     if (emitted_kernel.second->get_or_emit_source() == nullptr)
    //         NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be emitted before this pass:"
    //                                        << ins->getGNode()->get_name();
    //     kernel = emitted_kernel.second;

    //     return kernel;
    // };

    struct input_info
    {
        std::shared_ptr<nnfusion::graph::GNode> input_node;
        std::shared_ptr<descriptor::Tensor> tensor;
        size_t offset = 0;
    };
    std::unordered_map<std::shared_ptr<descriptor::Tensor>, struct input_info> inplace_inputs;
    std::unordered_map<std::shared_ptr<descriptor::Tensor>, size_t> inplace_use_count;
    std::unordered_map<std::shared_ptr<nnfusion::graph::GNode>, size_t> orders;
    size_t node_order = 0;
    std::unordered_map<std::shared_ptr<nnfusion::graph::GNode>, ir::Instruction::Pointer>
        node_to_ins;

    size_t annotate_concat = 0;
    size_t inplace_concat = 0;

    for (auto iterator : p)
    {
        for (auto ins : *iterator)
        {
            if (ins->name() == "Memcpy")
                continue;
            auto gnode = ins->getGNode();
            orders[gnode] = node_order++;
            node_to_ins[gnode] = ins;

            // skip parameter tensors.
            if (gnode->is_parameter())
                continue;

            // if (auto kernel = get_kernel(ins))
            if (auto kernel = ins->getKernel())
            {
                NNFUSION_CHECK_NOT_NULLPTR(kernel->m_context);

                if (auto annotations = kernel->m_context->annotations)
                {
                    // concat in_place_oi should be treated differently
                    if (std::dynamic_pointer_cast<nnfusion::op::Concat>(gnode->get_op_ptr()))
                    {
                        annotate_concat++;
                        NNFUSION_CHECK(annotations->get_in_place_oi_pairs().size() ==
                                       kernel->m_context->inputs.size());
                        std::vector<std::pair<std::shared_ptr<nnfusion::graph::GNode>,
                                              std::shared_ptr<descriptor::Tensor>>>
                            input_candidates;
                        size_t first_candidate = 0;
                        bool can_do_inplace_concat = true;
                        bool is_persistent = false;

                        for (auto oi_pair : annotations->get_in_place_oi_pairs())
                        {
                            //auto output = kernel->m_context->outputs[oi_pair.output];
                            auto input = kernel->m_context->inputs[oi_pair.input];
                            auto input_gnode = gnode->get_in_edge(oi_pair.input)->get_src();

                            if (input_gnode->is_parameter())
                            {
                                can_do_inplace_concat = false;
                                break;
                            }

                            if (inplace_inputs.count(input) > 0)
                            {
                                auto info = inplace_inputs[input];
                                if (info.offset > 0 || info.tensor->size() != input->size() ||
                                    info.tensor->is_parameter())
                                {
                                    can_do_inplace_concat = false;
                                    break;
                                }
                                input_gnode = info.input_node;
                                input = info.tensor;
                            }

                            auto ins = node_to_ins[input_gnode];
                            if ((*ins)["InplaceTensorMapping"].is_valid())
                            {
                                auto in_place_outputs =
                                    (*ins)["InplaceTensorMapping"]
                                        .as<std::map<std::shared_ptr<descriptor::Tensor>,
                                                     std::pair<std::shared_ptr<descriptor::Tensor>,
                                                               size_t>>>();
                                if (in_place_outputs.count(input) > 0)
                                {
                                    can_do_inplace_concat = false;
                                    break;
                                }
                            }

                            for (auto cand : input_candidates)
                            {
                                auto tensor_a = cand.second;
                                auto tensor_b = input;
                                if (inplace_inputs.count(tensor_a) > 0)
                                    tensor_a = inplace_inputs[tensor_a].tensor;
                                if (inplace_inputs.count(tensor_b) > 0)
                                    tensor_b = inplace_inputs[tensor_b].tensor;

                                // duplicated candidates
                                if (tensor_a == tensor_b)
                                {
                                    can_do_inplace_concat = false;
                                    break;
                                }
                            }
                            if (input->is_persistent())
                                is_persistent = true;

                            input_candidates.push_back({input_gnode, input});
                            if (orders[input_gnode] <
                                orders[input_candidates[first_candidate].first])
                            {
                                first_candidate = input_candidates.size() - 1;
                            }
                        }

                        if (can_do_inplace_concat)
                        {
                            inplace_concat++;
                            for (size_t i = 0; i < input_candidates.size(); i++)
                            {
                                auto oi_pair = annotations->get_in_place_oi_pairs()[i];
                                auto output = kernel->m_context->outputs[0];
                                auto input = input_candidates[i].second;
                                auto input_gnode = input_candidates[i].first;
                                auto input_ins = node_to_ins[input_gnode];

                                if (is_persistent)
                                {
                                    output->set_persistent();
                                    output->set_group("persist");
                                }

                                if (is_persistent)
                                {
                                    output->set_persistent();
                                    output->set_group("persist");
                                }

                                std::map<std::shared_ptr<descriptor::Tensor>,
                                         std::pair<std::shared_ptr<descriptor::Tensor>, size_t>>
                                    in_place_outputs;
                                if ((*input_ins)["InplaceTensorMapping"].is_valid())
                                {
                                    in_place_outputs =
                                        (*input_ins)["InplaceTensorMapping"]
                                            .as<std::map<
                                                std::shared_ptr<descriptor::Tensor>,
                                                std::pair<std::shared_ptr<descriptor::Tensor>,
                                                          size_t>>>();
                                    NNFUSION_CHECK(in_place_outputs.count(input) == 0);
                                }
                                in_place_outputs.insert(
                                    {input, std::make_pair(output, oi_pair.input_offset)});

                                (*input_ins)["InplaceTensorMapping"] = in_place_outputs;

                                inplace_inputs[input].input_node = gnode;
                                inplace_inputs[input].tensor = output;
                                inplace_inputs[input].offset = oi_pair.input_offset;
                                if (inplace_use_count.count(output) == 0)
                                    inplace_use_count[output] = 0;
                                inplace_use_count[output] += 1;

                                if (inplace_use_count[input] > 0)
                                {
                                    // update tensors who refer to this input
                                    for (auto oi : inplace_inputs)
                                    {
                                        if (oi.second.tensor == input)
                                        {
                                            auto offset = oi.second.offset;
                                            inplace_inputs[oi.first] = inplace_inputs[input];
                                            inplace_inputs[oi.first].offset += offset;
                                        }
                                    }
                                    inplace_use_count[output] += inplace_use_count[input];
                                    inplace_use_count[input] = 0;
                                }

                                // move the allocation of concat output tensor to the first input node
                                if (i == first_candidate)
                                {
                                    NNFUSION_CHECK(input_ins->liveness_new_list.count(output) == 0);
                                    NNFUSION_CHECK(ins->liveness_new_list.count(output) > 0);
                                    input_ins->liveness_new_list.insert(output);
                                    ins->liveness_new_list.erase(output);
                                }
                            }
                        }
                    }
                    else
                    {
                        std::map<std::shared_ptr<descriptor::Tensor>,
                                 std::pair<std::shared_ptr<descriptor::Tensor>, size_t>>
                            in_place_outputs;
                        for (auto oi_pair : annotations->get_in_place_oi_pairs())
                        {
                            auto output = kernel->m_context->outputs[oi_pair.output];
                            auto input = kernel->m_context->inputs[oi_pair.input];
                            auto input_gnode = gnode->get_in_edge(oi_pair.input)->get_src();

                            if (!is_same_dev(input, output))
                            {
                                NNFUSION_LOG(NNFUSION_WARNING)
                                    << "Tensor inplace oi pairs are not in same device, ignored.";
                                continue;
                            }

                            if (input_gnode->is_parameter() && !oi_pair.force_inplace)
                            {
                                continue;
                            }

                            // skip pair with constant output tensor, as this might be used by runtime constant folding
                            if (output->is_constant())
                            {
                                continue;
                            }

                            // If the inplace is destructive, the output should not overwrite the constant tensor,
                            // parameter tensor and persistent tensor, and the input must be in free_list of this node.
                            // Otherwise, it is safe to do inplace reuse.
                            if (ins->liveness_new_list.count(output) != 0)
                            {
                                if (oi_pair.force_inplace || !oi_pair.destructive ||
                                    ///\todo uncomment below after correctness
                                    // if ((!oi_pair.destructive && !output->is_persistent()) ||
                                    (!input_gnode->is_constant() && !input->is_persistent() &&
                                     (inplace_inputs.count(input) == 0 ||
                                      inplace_use_count[inplace_inputs[input].tensor] == 0) &&
                                     ins->liveness_free_list.count(input) != 0))
                                {
                                    in_place_outputs.insert(
                                        {output, std::make_pair(input, oi_pair.input_offset)});

                                    if (inplace_inputs.count(input) > 0)
                                    {
                                        inplace_inputs[output] = inplace_inputs[input];
                                        inplace_inputs[output].offset += oi_pair.input_offset;
                                    }
                                    else
                                    {
                                        inplace_inputs[output].input_node = input_gnode;
                                        inplace_inputs[output].tensor = input;
                                        inplace_inputs[output].offset = oi_pair.input_offset;
                                    }
                                    auto input_tensor = inplace_inputs[output].tensor;
                                    if (inplace_use_count.count(input_tensor) == 0)
                                        inplace_use_count[input_tensor] = 0;
                                    inplace_use_count[input_tensor] += 1;
                                }
                            }
                        }

                        if (in_place_outputs.size() > 0)
                        {
                            (*ins)["InplaceTensorMapping"] = in_place_outputs;
                        }
                    }
                }

                // decrease inplace_use_count
                for (std::shared_ptr<descriptor::Tensor> tensor : ins->liveness_free_list)
                {
                    if (inplace_inputs.count(tensor) > 0)
                    {
                        auto input_tensor = inplace_inputs[tensor].tensor;
                        NNFUSION_CHECK(inplace_use_count.count(input_tensor) > 0);
                        NNFUSION_CHECK(inplace_use_count[input_tensor] > 0);
                        inplace_use_count[input_tensor]--;
                    }
                }
            }
        }
    }

    NNFUSION_LOG(INFO) << "Inpalce tensor analysis: annotated concat: " << annotate_concat
                       << ", inlace_concat: " << inplace_concat;

    return true;
}
