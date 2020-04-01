// Microsoft (c) 2019, NNFUSION TEAM
#include "tensor_memory_layout.hpp"

#include <exception>
#include <queue>
#include <sstream>
#include <utility>

#include "nnfusion/common/util.hpp"
#include "nnfusion/engine/memory_allocator.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

#include "nnfusion/core/operators/op_define/concat.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"

using namespace std;
using namespace nnfusion;
using namespace nnfusion::pass;
using namespace nnfusion::kernels;

DEFINE_bool(fmem_trace, false, "Record and dump memory trace.");
DEFINE_string(fmem_log_path, "memory.log", "The file path of memory log.");

bool AssignTensorMemoryLayout::run(std::shared_ptr<InterpreterContext> ctx,
                                   std::shared_ptr<TranslationUnit> tu)
{
    bool dump_trace = FLAGS_fmem_trace;
    string mem_log_path = FLAGS_fmem_log_path;

    // Open memory log file.
    std::ofstream mem_log;
    if (dump_trace)
        mem_log.open(mem_log_path);

    MemoryAllocatorFactory maf(m_alignment, m_disable_memory_sharing);

    auto is_same_dev = [](shared_ptr<const descriptor::Tensor> a,
                          shared_ptr<const descriptor::Tensor> b) {
        return (a->get_device_type() == b->get_device_type()) &&
               (a->get_device_id() == b->get_device_id());
    };

    std::unordered_set<shared_ptr<descriptor::Tensor>> persistent_tensors;
    auto& p = tu->program;

    for (auto iterator : p)
    {
        for (auto ins : *iterator)
        {
            auto gnode = ins->getGNode();
            // do not allocate parameter tensors.
            if (gnode->get_op_ptr()->is_parameter())
                continue;
            auto emitted_kernels = (*ins)["Kernel_Selection_Result"]
                                       .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();
            auto emitter_iter =
                find_if(emitted_kernels.begin(),
                        emitted_kernels.end(),
                        [this](pair<DeviceType, KernelEmitter::Pointer>& i) {
                            return (i.first == CUDA_GPU || i.first == DeviceType::ROCM_GPU);
                        });

            KernelEmitter::Pointer kernel = nullptr;

            if (emitter_iter == emitted_kernels.end() || emitter_iter->second == nullptr)
                // Can assign tensor layout even kernel is not emitted.
                LOG(WARNING) << "Kernel should be emitted before this pass:" << gnode->get_name();
            else
                kernel = emitter_iter->second;
            // Tensors should be considered
            // Node: inputs outputs
            // Kernel Context: +tensors

            std::map<std::shared_ptr<descriptor::Tensor>, std::shared_ptr<descriptor::Tensor>>
                in_place_outputs;
            std::set<std::shared_ptr<descriptor::Tensor>> reused_inputs;
            std::unordered_set<std::shared_ptr<descriptor::Tensor>> alloc_temp;

            if (kernel != nullptr)
            {
                CHECK_NOT_NULLPTR(kernel->m_context);
                // Allocate NoneResuseable Space for Persistent Tensors
                for (size_t i = 0; i < kernel->m_context->tensors.size(); i++)
                {
                    auto tensor = kernel->m_context->tensors[i];
                    if (!tensor->is_persistent())
                        alloc_temp.insert(tensor);
                }

                // concat in_place_oi should be treated differently
                if (!std::dynamic_pointer_cast<nnfusion::op::Concat>(gnode->get_op_ptr()))
                {
                    if (auto annotations = kernel->m_context->annotations)
                    {
                        for (auto oi_pair : annotations->get_in_place_oi_pairs())
                        {
                            auto output = kernel->m_context->outputs[oi_pair.output];
                            auto input = kernel->m_context->inputs[oi_pair.input];
                            auto input_gnode = gnode->get_in_edge(oi_pair.input)->get_src();

                            //should not overwrite constant tensor and parameter tensor
                            if (input_gnode->get_op_ptr()->is_parameter() ||
                                input_gnode->get_op_ptr()->is_constant())
                                continue;

                            if (!is_same_dev(input, output))
                            {
                                LOG(WARNING)
                                    << "Tensor inplace oi pairs are not in same device, ignored.";
                                continue;
                            }

                            // memory of persistent tensors should not be reused.
                            if (gnode->liveness_free_list.count(input) != 0 &&
                                gnode->liveness_new_list.count(output) != 0 &&
                                !input->is_persistent())
                            {
                                in_place_outputs.insert({output, input});
                                reused_inputs.insert(input);
                            }
                        }
                    }
                }
            }

            unordered_set<std::shared_ptr<descriptor::Tensor>> newlist(alloc_temp);
            // The output of output nodes refers to the input, so there is NO need
            // to allocate memory space for output of output nodes.
            if (!gnode->get_op_ptr()->is_output())
                newlist.insert(gnode->liveness_new_list.begin(), gnode->liveness_new_list.end());
            for (std::shared_ptr<descriptor::Tensor> tensor : newlist)
            {
                if (!tensor->is_persistent())
                {
                    auto allocator = maf.get_allocator(tensor);
                    if (in_place_outputs.count(tensor))
                    {
                        size_t offset = in_place_outputs.at(tensor)->get_pool_offset();
                        allocator->allocate(tensor, offset);
                    }
                    else
                    {
                        allocator->allocate(tensor);
                    }
                }
                else
                {
                    persistent_tensors.insert(tensor);
                }
            }

            if (!m_disable_memory_sharing)
            {
                unordered_set<shared_ptr<descriptor::Tensor>> freelist(alloc_temp);
                freelist.insert(gnode->liveness_free_list.begin(), gnode->liveness_free_list.end());
                for (std::shared_ptr<descriptor::Tensor> tensor : freelist)
                {
                    if (reused_inputs.count(tensor) == 0 && !tensor->is_persistent() &&
                        !tensor->is_parameter())
                    {
                        auto allocator = maf.get_allocator(tensor);
                        allocator->free(tensor);
                    }
                }
            }
            //dump memory trace at the time scale of node.
            if (dump_trace)
            {
                mem_log << gnode->get_name() << "\n";
                for (const auto& allocator : MemoryAllocatorFactory::get_allocator_list())
                {
                    allocator.second->dump(mem_log);
                }
                mem_log << "\n";
            }
        }
    }
    // allocate persistent tensors with NO_REUSE scheme.
    for (auto tensor : persistent_tensors)
    {
        auto allocator = maf.get_allocator(tensor);
        allocator->set_alloc_scheme(nnfusion::MemoryAllocator::allocation_scheme::NO_REUSE);
        allocator->allocate(tensor);
    }

    if (dump_trace)
    {
        mem_log << "----allocate persistent tensors----\n";
        for (const auto& allocator : MemoryAllocatorFactory::get_allocator_list())
        {
            allocator.second->dump(mem_log);
        }
        // close memory log file.
        mem_log.close();
    }
    return true;
}
