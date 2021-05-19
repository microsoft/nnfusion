// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "liveness_analysis.hpp"

#include <exception>
#include <queue>
#include <sstream>
#include <utility>

#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/engine/async_manager.hpp"

using namespace std;
using namespace nnfusion;
using namespace nnfusion::pass;
using namespace nnfusion::kernels;
using namespace nnfusion::async;

DEFINE_bool(frt_const_folding, false, "Add runtime constant folding.");

bool TensorLivenessAnalysis::run(std::shared_ptr<InterpreterContext> ctx,
                                 std::shared_ptr<TranslationUnit> tu)
{
    bool enable_rt_const_folding = FLAGS_frt_const_folding;
    std::unordered_set<shared_ptr<descriptor::Tensor>> persist_candidate;

    auto& p = tu->program;
    for (auto block_iter : p)
    {
        for (auto ins : *block_iter)
        {
            auto gnode = ins->getGNode();
            if (!(*ins)["Async_info"].is_valid())
            {
                string name = gnode ? gnode->get_name() : ins->name();
                NNFUSION_CHECK_FAIL() << "Async info should be assigned before this pass：" << name;
            }
            auto& async_info = (*ins)["Async_info"].as<AsyncExecutionInfo>();
            std::shared_ptr<nnfusion::async::Stream> stream;

            if (async_info.execution_stream != nullptr)
                stream = async_info.execution_stream;
            else
                stream = async_info.execution_thread;

            auto stream_id = stream->get_stream_id();
            if (gnode && gnode->is_parameter())
            {
                auto& outputs = ins->get_outputs();
                for (size_t i = 0; i < outputs.size(); i++)
                {
                    auto tensor = outputs[i];
                    tensor->set_parameter();
                    tensor->set_persistent();
                    set_tensor_group(tensor, to_string(stream_id));
                }
            }
            else if (gnode && gnode->is_variable())
            {
                auto& outputs = ins->get_outputs();
                for (size_t i = 0; i < outputs.size(); i++)
                {
                    auto tensor = outputs[i];
                    tensor->set_persistent();
                    set_tensor_group(tensor, to_string(stream_id));
                }
            }
            else if (gnode && gnode->get_op_ptr()->is_output())
            {
                auto& outputs = ins->get_outputs();
                for (size_t i = 0; i < outputs.size(); i++)
                {
                    auto tensor = outputs[i];
                    tensor->set_persistent();
                    set_tensor_group(tensor, to_string(stream_id));
                }
                auto& inputs = ins->get_inputs();
                for (size_t i = 0; i < inputs.size(); i++)
                {
                    auto tensor = inputs[i];
                    tensor->set_persistent();
                    set_tensor_group(tensor, to_string(stream_id));
                }
            }
            else if (gnode && gnode->is_constant())
            {
                auto& outputs = ins->get_outputs();
                for (size_t i = 0; i < outputs.size(); i++)
                {
                    auto tensor = outputs[i];
                    if (enable_rt_const_folding)
                    {
                        persist_candidate.insert(tensor);
                    }
                    else
                    {
                        tensor->set_persistent();
                    }
                    set_tensor_group(tensor, to_string(stream_id));
                }
            }
            else
            {
                // add cross_stream tensor
                auto& inputs = ins->get_inputs();
                for (size_t i = 0; i < inputs.size(); i++)
                {
                    auto tensor = inputs[i];
                    set_tensor_group(tensor, to_string(stream_id));
                }
                // set output tensor's group id
                auto& outputs = ins->get_outputs();
                for (size_t i = 0; i < outputs.size(); i++)
                {
                    auto tensor = outputs[i];
                    set_tensor_group(tensor, to_string(stream_id));
                }
                // set temp tensor's group id
                auto& tensors = ins->get_internal_tensors();
                for (size_t i = 0; i < tensors.size(); i++)
                {
                    auto tensor = tensors[i];
                    set_tensor_group(tensor, to_string(stream_id));
                }
            }
        }
    }

    if (enable_rt_const_folding)
    {
        for (auto block_iter : p)
        {
            for (auto ins : *block_iter)
            {
                auto gnode = ins->getGNode();
                if ((*ins)["rt_const_folding"].is_valid_as<bool>())
                {
                    auto& outputs = ins->get_outputs();
                    for (size_t i = 0; i < outputs.size(); i++)
                    {
                        auto tensor = outputs[i];
                        persist_candidate.insert(tensor);
                    }
                }
                else
                {
                    bool has_const = false;
                    bool has_non_const = false;
                    std::unordered_set<shared_ptr<descriptor::Tensor>> tmp;
                    auto& inputs = ins->get_inputs();
                    for (size_t i = 0; i < inputs.size(); i++)
                    {
                        auto tensor = inputs[i];
                        if (persist_candidate.find(tensor) != persist_candidate.end())
                        {
                            tmp.insert(tensor);
                            has_const = true;
                        }
                        else
                        {
                            has_non_const = true;
                        }
                    }
                    if (has_const && has_non_const)
                    {
                        for (auto tensor : tmp)
                        {
                            tensor->set_persistent();
                            tensor->set_constant();
                            tensor->set_group("persist");
                        }
                    }
                }
            }
        }
    }

    std::unordered_set<shared_ptr<descriptor::Tensor>> currently_live;

    // traverse instructions in reverse order
    for (auto block_it = p.rbegin(); block_it != p.rend(); block_it++)
    {
        auto block_p = *block_it;
        for (auto ins_it = block_p->rbegin(); ins_it != block_p->rend(); ins_it++)
        {
            auto ins = *ins_it;
            ins->liveness_new_list.clear();
            ins->liveness_free_list.clear();

            std::unordered_set<std::shared_ptr<descriptor::Tensor>> input_tensor_decls;
            std::unordered_set<std::shared_ptr<descriptor::Tensor>> output_tensor_decls;
            auto& inputs = ins->get_inputs();
            for (size_t i = 0; i < inputs.size(); i++)
            {
                auto tensor = inputs[i];
                input_tensor_decls.insert(tensor);
            }
            auto& outputs = ins->get_outputs();
            for (size_t i = 0; i < outputs.size(); i++)
            {
                auto tensor = outputs[i];
                output_tensor_decls.insert(tensor);
            }

            std::unordered_set<std::shared_ptr<descriptor::Tensor>> free_tensor_decls;
            std::unordered_set<std::shared_ptr<descriptor::Tensor>> new_tensor_decls;
            std::unordered_set<std::shared_ptr<descriptor::Tensor>> all_tensor_decls =
                input_tensor_decls;
            all_tensor_decls.insert(output_tensor_decls.begin(), output_tensor_decls.end());

            for (std::shared_ptr<descriptor::Tensor> tensor_decl : all_tensor_decls)
            {
                if (currently_live.find(tensor_decl) == currently_live.end())
                {
                    // this is the last node that value is seen in
                    // delete it at the end of the op
                    currently_live.insert(tensor_decl);
                    if (cross_stream.find(tensor_decl) == cross_stream.end())
                        free_tensor_decls.insert(tensor_decl);
                }
            }

            for (std::shared_ptr<descriptor::Tensor> output_decl : output_tensor_decls)
            {
                auto currently_live_it = currently_live.find(output_decl);
                if (currently_live_it != currently_live.end())
                {
                    new_tensor_decls.insert(output_decl);
                    currently_live.erase(currently_live_it);
                }
            }
            ins->liveness_free_list = free_tensor_decls;
            ins->liveness_new_list = new_tensor_decls;
        }
    }

    NNFUSION_LOG(INFO) << "------------------Liveness analysis pass done.";
    return true;
}
void TensorLivenessAnalysis::set_tensor_group(shared_ptr<descriptor::Tensor> tensor,
                                              const std::string& group)
{
    if (tensor->get_group() == "")
    {
        if (tensor->is_persistent())
        {
            tensor->set_group("persist");
        }
        else
        {
            tensor->set_group(group);
        }
    }
    else if (tensor->get_group() != group)
    {
        cross_stream.insert(tensor);
    }
}