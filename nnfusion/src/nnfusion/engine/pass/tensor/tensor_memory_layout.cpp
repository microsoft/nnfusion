// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "tensor_memory_layout.hpp"

#include <exception>
#include <queue>
#include <sstream>
#include <utility>

#include "nnfusion/common/util.hpp"
#include "nnfusion/engine/memory_allocator.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

#include "nnfusion/core/operators/op_define/concat.hpp"

using namespace std;
using namespace nnfusion;
using namespace nnfusion::pass;
using namespace nnfusion::kernels;

DEFINE_bool(fmem_trace, false, "Record and dump memory trace.");
DEFINE_string(fmem_log_path, "memory.log", "The file path of memory log.");
DECLARE_string(fhlsl_codegen_type);
DECLARE_bool(fextern_result_memory);
DECLARE_bool(fhost_entry);

bool AssignTensorMemoryLayout::run(std::shared_ptr<InterpreterContext> ctx,
                                   std::shared_ptr<TranslationUnit> tu)
{
    bool dump_trace = FLAGS_fmem_trace;
    string mem_log_path = tu->m_graph->get_name() + "_" + FLAGS_fmem_log_path;

    // Open memory log file.
    std::ofstream mem_log;
    if (dump_trace)
        mem_log.open(mem_log_path);

    NNFUSION_CHECK(tu->memory_allocator_factory == nullptr);
    tu->memory_allocator_factory =
        std::make_shared<MemoryAllocatorFactory>(m_alignment, m_disable_memory_sharing);
    auto maf = tu->memory_allocator_factory;
    // std::unordered_set<shared_ptr<descriptor::Tensor>> persistent_tensors;
    auto& p = tu->program;

    for (auto iterator : p)
    {
        for (auto ins : *iterator)
        {
            MemoryInfo mem_info;
            auto gnode = ins->getGNode();
            // do not allocate parameter tensors.
            if (gnode && gnode->is_parameter() && !FLAGS_fhost_entry)
                continue;
            // Tensors should be considered
            // Node: inputs outputs
            // Kernel Context: +tensors
            // <output, <input, offset>>
            std::map<std::shared_ptr<descriptor::Tensor>,
                     std::pair<std::shared_ptr<descriptor::Tensor>, size_t>>
                in_place_outputs;
            std::unordered_set<std::shared_ptr<descriptor::Tensor>> alloc_temp;
            if (auto kernel = ins->getKernel())
            {
                NNFUSION_CHECK_NOT_NULLPTR(kernel->m_context);
                // Allocate temp tensors
                for (size_t i = 0; i < kernel->m_context->tensors.size(); i++)
                {
                    auto tensor = kernel->m_context->tensors[i];
                    // NNFUSION_CHECK(!tensor->is_persistent());
                    alloc_temp.insert(tensor);
                }

                if ((*ins)["InplaceTensorMapping"].is_valid())
                {
                    in_place_outputs =
                        (*ins)["InplaceTensorMapping"]
                            .as<std::map<std::shared_ptr<descriptor::Tensor>,
                                         std::pair<std::shared_ptr<descriptor::Tensor>, size_t>>>();
                }
            }

            unordered_set<std::shared_ptr<descriptor::Tensor>> newlist(alloc_temp);
            // todo: this hack is to eliminate d2d copy caused by extern result memory
            bool skip = false;
            if (FLAGS_fextern_result_memory && gnode)
            {
                bool all_users_are_result = true;
                for (size_t i = 0; i < gnode->get_out_edges().size(); i++)
                {
                    auto dst = gnode->get_out_edges()[i]->get_dst();

                    if (dst && !dst->get_op_ptr()->is_output())
                    {
                        all_users_are_result = false;
                        break;
                    }
                }
                if (all_users_are_result)
                {
                    skip = true;
                }
            }
            // The output of output nodes refers to the input, so there is NO need
            // to allocate memory space for output of output nodes.
            if (!skip && (!gnode || !gnode->get_op_ptr()->is_output() ||
                          (gnode->get_op_ptr()->is_output() && !FLAGS_fextern_result_memory)))
                newlist.insert(ins->liveness_new_list.begin(), ins->liveness_new_list.end());

            // Allocate in two passes to make sure ref-tensors is after non-ref-tensors
            std::vector<std::shared_ptr<descriptor::Tensor>> ref_tensors;
            for (std::shared_ptr<descriptor::Tensor> tensor : newlist)
            {
                if (in_place_outputs.count(tensor))
                {
                    ref_tensors.push_back(tensor);
                    mem_info.alloc_ref.push_back(tensor);
                    auto root = tensor->get_root_tensor();
                }
                else
                {
                    auto allocator = maf->get_allocator(tensor);
                    tensor->set_pool(allocator->get_name());
                    allocator->allocate(tensor);
                    mem_info.alloc_new.push_back(tensor);
                }
            }

            for (std::shared_ptr<descriptor::Tensor> tensor : ref_tensors)
            {
                NNFUSION_CHECK(in_place_outputs.count(tensor) > 0);
                auto parent_tensor = in_place_outputs.at(tensor).first;
                size_t tensor_offset = in_place_outputs.at(tensor).second;

                auto root_tensor = parent_tensor->get_root_tensor()
                                       ? parent_tensor->get_root_tensor()
                                       : parent_tensor;
                auto allocator = maf->get_allocator(root_tensor);
                allocator->allocate(
                    tensor, root_tensor, parent_tensor->get_pool_offset() + tensor_offset);
            }

            if (!m_disable_memory_sharing)
            {
                unordered_set<shared_ptr<descriptor::Tensor>> freelist(alloc_temp);
                freelist.insert(ins->liveness_free_list.begin(), ins->liveness_free_list.end());
                for (std::shared_ptr<descriptor::Tensor> tensor : freelist)
                {
                    // persistent tensor will not be reused
                    auto root_tensor = tensor->get_root_tensor();
                    if ((!root_tensor && !tensor->is_persistent() && !tensor->is_parameter()) ||
                        (root_tensor && !root_tensor->is_persistent() &&
                         !root_tensor->is_parameter()))
                    {
                        // auto root_tensor = tensor->get_root_tensor();
                        auto allocator = maf->get_allocator(root_tensor ? root_tensor : tensor);
                        allocator->free(tensor);
                        mem_info.free.push_back(root_tensor ? root_tensor : tensor);
                    }
                }
            }
            //dump memory trace at the time scale of node.
            if (dump_trace)
            {
                string name = gnode ? gnode->get_name() : ins->name();
                mem_log << name << "\n";
                for (const auto& allocator : maf->get_allocator_list())
                {
                    allocator.second->dump(mem_log);
                }
                mem_log << "\n";
            }

            (*ins)["MemoryInfo"] = mem_info;
        }
    }

    if (dump_trace)
    {
        // close memory log file.
        mem_log.close();
    }
    NNFUSION_LOG(INFO) << "---------------Tensor memory layout pass done.";
    return true;
}
