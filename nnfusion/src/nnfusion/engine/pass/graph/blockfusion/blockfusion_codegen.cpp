// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "blockfusion_codegen.hpp"
#include "nnfusion/core/kernels/common_langunit.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/engine/async_manager.hpp"

using namespace nnfusion;
using namespace nnfusion::blockfusion;
using namespace nnfusion::kernels;

using namespace std;

size_t BlockFusionCudaCodegen::unique_func_id = 0;

BlockFusionCudaCodegen::BlockFusionCudaCodegen(shared_ptr<KernelContext> ctx,
                                               const BlockExecutorProgram& _block_executor_program,
                                               int _codegen_opt_level)
    : CudaEmitter(ctx)
    , block_executor_program(_block_executor_program)
{
    NNFUSION_CHECK_NOT_NULLPTR(FuseContext());

    // control whether dedupe block_kernels
    this->is_dedupe_block_kernels = true;
    // control whether optimize instruction_step_to in codegen
    this->is_ins_step_to_opt = true;
    this->codegen_opt_level = _codegen_opt_level;

    deduped_kernel_id_map.clear();
    for (int kernel_id = 0; kernel_id < block_executor_program.block_kernels.size(); kernel_id++)
    {
        deduped_kernel_id_map[kernel_id] = kernel_id;
    }

    // dedupe block_kernels
    // key: signature (data type) + device_function_body (kernel code)
    if (this->is_dedupe_block_kernels == true)
    {
        for (int kernel_id = 0; kernel_id < block_executor_program.block_kernels.size();
             kernel_id++)
        {
            std::string block_kernel_key = block_executor_program.block_kernels[kernel_id]
                                               ->get_or_emit_source()
                                               ->signature_unit->get_code() +
                                           block_executor_program.block_kernels[kernel_id]
                                               ->emit_device_function_body()
                                               ->get_code();
            for (int deduped_kernel_id = 0; deduped_kernel_id < kernel_id; deduped_kernel_id++)
            {
                std::string deduped_kernel_key =
                    block_executor_program.block_kernels[deduped_kernel_id]
                        ->get_or_emit_source()
                        ->signature_unit->get_code() +
                    block_executor_program.block_kernels[deduped_kernel_id]
                        ->emit_device_function_body()
                        ->get_code();
                if (block_kernel_key == deduped_kernel_key)
                {
                    deduped_kernel_id_map[kernel_id] = deduped_kernel_id;
                    break;
                }
            }
        }
    }

    // optimize instruction_step_to in codegen
    if (this->is_ins_step_to_opt == true)
    {
        for (auto be : block_executor_program.block_executor_instructions)
        {
            if (be.size() <= 1)
            {
                continue;
            }
            for (int ins_id = 0; ins_id < be.size() - 1; ins_id++)
            {
                auto ins_cur =
                    std::dynamic_pointer_cast<BlockExecutorInstructionStepTo>(be[ins_id]);
                auto ins_next =
                    std::dynamic_pointer_cast<BlockExecutorInstructionStepTo>(be[ins_id + 1]);
                if (ins_cur != nullptr && ins_next != nullptr)
                {
                    NNFUSION_CHECK(ins_cur->step_id <= ins_next->step_id);
                    be.erase(be.begin() + ins_id);
                }
            }
        }
    }
}

std::shared_ptr<KernelContext> BlockFusionCudaCodegen::FuseContext()
{
    std::shared_ptr<KernelContext> ctx = this->m_context;

    ctx->kernels.clear();
    for (auto block_kernel : block_executor_program.block_kernels)
    {
        auto kernel = std::dynamic_pointer_cast<KernelEmitter>(block_kernel);
        NNFUSION_CHECK_NOT_NULLPTR(kernel);
        ctx->kernels.push_back(kernel);
    }

    std::unordered_map<std::string, size_t> node_inputs;
    std::unordered_map<std::string, size_t> node_outputs;
    std::unordered_map<std::string, size_t> node_temps;
    std::unordered_set<int64_t> nodes_in_group;
    std::map<std::string, size_t> fused_tensor_input_map;  // tensor_name -> fused_ctx input id
    std::map<std::string, size_t> fused_tensor_output_map; // tensor_name -> fused_ctx output id
    for (auto kernel_emitter : ctx->kernels)
    {
        auto gnode = kernel_emitter->m_context->gnode;
        nodes_in_group.insert(gnode->get_id());
    }
    // process input and output tensors of this fusion group
    for (auto kernel_emitter : ctx->kernels)
    {
        auto gnode = kernel_emitter->m_context->gnode;
        for (const auto& in_edge : gnode->get_in_edges())
        {
            if (!in_edge->is_control_edge() &&
                nodes_in_group.find(in_edge->get_src()->get_id()) == nodes_in_group.end())
            {
                auto tv = gnode->get_input_tensor_ptr(in_edge->get_dst_input());
                NNFUSION_CHECK_NOT_NULLPTR(tv);
                if (node_inputs.find(tv->get_name()) == node_inputs.end())
                {
                    node_inputs[tv->get_name()] = 1;
                    ctx->inputs.push_back(tv);
                    ctx->input_names.push_back(tv->get_name());
                    fused_tensor_input_map[tv->get_name()] = ctx->inputs.size() - 1;
                }
                else
                {
                    node_inputs[tv->get_name()]++;
                }
            }
        }

        for (const auto& out_edge : gnode->get_out_edges())
        {
            if (!out_edge->is_control_edge() &&
                nodes_in_group.find(out_edge->get_dst()->get_id()) == nodes_in_group.end())
            {
                auto tv = gnode->get_output_tensor_ptr(out_edge->get_src_output());
                NNFUSION_CHECK_NOT_NULLPTR(tv);
                if (node_outputs.find(tv->get_name()) == node_outputs.end())
                {
                    node_outputs[tv->get_name()] = 1;
                    ctx->outputs.push_back(tv);
                    ctx->output_names.push_back(tv->get_name());
                    fused_tensor_output_map[tv->get_name()] = ctx->outputs.size() - 1;
                }
                else
                {
                    node_outputs[tv->get_name()]++;
                }
            }
        }
    }
    // process internal tensors
    for (auto kernel_emitter : ctx->kernels)
    {
        auto gnode = kernel_emitter->m_context->gnode;
        for (size_t i = 0; i < gnode->get_input_size(); i++)
        {
            auto tv = gnode->get_input_tensor_ptr(i);
            NNFUSION_CHECK_NOT_NULLPTR(tv);

            if (node_inputs.find(tv->get_name()) == node_inputs.end())
            {
                if (node_temps.find(tv->get_name()) == node_temps.end())
                {
                    node_temps[tv->get_name()] = 1;
                    ctx->tensors.push_back(tv);
                    ctx->tensor_names.push_back(tv->get_name());
                }
                else
                {
                    node_temps[tv->get_name()]++;
                }
            }
        }
        for (size_t i = 0; i < gnode->get_output_size(); i++)
        {
            auto tv = gnode->get_output_tensor_ptr(i);
            NNFUSION_CHECK_NOT_NULLPTR(tv);

            if (node_outputs.find(tv->get_name()) == node_outputs.end())
            {
                if (node_temps.find(tv->get_name()) == node_temps.end())
                {
                    node_temps[tv->get_name()] = 1;
                    ctx->tensors.push_back(tv);
                    ctx->tensor_names.push_back(tv->get_name());
                }
                else
                {
                    node_temps[tv->get_name()]++;
                }
            }
        }
    }

    // process inplace annotation
    for (auto kernel_emitter : ctx->kernels)
    {
        if (kernel_emitter->m_context->annotations != nullptr)
        {
            auto annotations = kernel_emitter->m_context->annotations->get_in_place_oi_pairs();
            for (auto annotation : annotations)
            {
                auto input_name = kernel_emitter->m_context->inputs[annotation.input]->get_name();
                auto output_name =
                    kernel_emitter->m_context->outputs[annotation.output]->get_name();
                if (fused_tensor_input_map.find(input_name) != fused_tensor_input_map.end() &&
                    fused_tensor_output_map.find(output_name) != fused_tensor_output_map.end())
                {
                    auto input_id = fused_tensor_input_map[input_name];
                    auto output_id = fused_tensor_output_map[output_name];
                    if (!ctx->annotations)
                    {
                        ctx->annotations = std::make_shared<Annotations>();
                    }
                    ctx->annotations->add_in_place_oi_pair({output_id,
                                                            input_id,
                                                            annotation.destructive,
                                                            annotation.input_offset,
                                                            annotation.force_inplace});
                }
            }
        }
    }

    for (auto arg : ctx->inputs)
    {
        ctx->dtypes.push_back(arg->get_element_type().c_type_string());
    }
    for (auto out : ctx->outputs)
    {
        ctx->dtypes.push_back(out->get_element_type().c_type_string());
    }
    for (auto temp : ctx->tensors)
    {
        ctx->dtypes.push_back(temp->get_element_type().c_type_string());
    }

    // convert tensor name to args
    all_args.clear();
    for (int i = 0; i < m_context->inputs.size(); i++)
    {
        auto& tensor = m_context->inputs[i];
        all_args[tensor->get_name()] = "input" + std::to_string(i);
    }
    for (int i = 0; i < m_context->outputs.size(); i++)
    {
        auto& tensor = m_context->outputs[i];
        all_args[tensor->get_name()] = "output" + std::to_string(i);
    }
    for (int i = 0; i < m_context->tensors.size(); i++)
    {
        auto& tensor = m_context->tensors[i];
        all_args[tensor->get_name()] = "temp" + std::to_string(i);
    }

    // if be_program has group_sync instructions (step_to, wait_for), set is_group_sync=true
    this->is_group_sync = false;
    for (auto be : block_executor_program.block_executor_instructions)
    {
        for (auto be_instruction : be)
        {
            if ((nullptr !=
                 std::dynamic_pointer_cast<BlockExecutorInstructionStepTo>(be_instruction)) ||
                (nullptr !=
                 std::dynamic_pointer_cast<BlockExecutorInstructionWaitFor>(be_instruction)))
            {
                this->is_group_sync = true;
                break;
            }
        }
        if (this->is_group_sync)
        {
            break;
        }
    }

    set_launch_config();

    // allocate be_state_buffer for group_sync
    if (this->is_group_sync)
    {
        dim3 grid_dim = this->get_grid_dim();
        std::shared_ptr<nnfusion::descriptor::Tensor> be_state_buffer(
            new nnfusion::descriptor::Tensor(
                nnfusion::element::i32,
                nnfusion::PartialShape(
                    {(size_t)grid_dim.x * (size_t)grid_dim.y * (size_t)grid_dim.z}),
                "BlockFusionKernel_" + std::to_string(BlockFusionCudaCodegen::unique_func_id) +
                    "_be_state_buffer",
                nnfusion::NNFusion_DeviceType::CUDA_GPU));

        be_state_buffer->set_memset(true, 0);
        be_state_buffer->set_persistent(true);

        ctx->tensors.push_back(be_state_buffer);
        ctx->tensor_names.push_back(be_state_buffer->get_name());

        ctx->dtypes.push_back(be_state_buffer->get_element_type().c_type_string());
    }

    return ctx;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_step_to_device_function()
{
    LanguageUnit_p _lu(new LanguageUnit("declaration::BlockFusion_step_to_device_function"));
    auto& lu = *_lu;

    // lu << "__device__ void BlockFusion_group_sync_init_device_function(volatile int* "
    //       "be_state_buffer)\n";
    // lu.block_begin();
    // lu << "if (threadIdx.x == 0)\n";
    // lu.block_begin();
    // lu << "be_state_buffer[blockIdx.x] = 0;\n";
    // lu.block_end();
    // lu.block_end();

    lu << "__device__ __forceinline__ void BlockFusion_step_to_device_function(volatile int* "
          "be_state_buffer, int be_id, int step_id)\n";
    lu.block_begin();
    lu << "if (threadIdx.x == 0)\n";
    lu.block_begin();
    lu << "be_state_buffer[be_id] = step_id;\n";
    lu.block_end();
    lu.block_end();

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_block_kernel_functions()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_block_kernel_function"));
    LanguageUnit& lu = *_lu;

    lu << "\n";

    if (this->is_dedupe_block_kernels == true)
    {
        for (int kernel_id = 0; kernel_id < block_executor_program.block_kernels.size();
             kernel_id++)
        {
            if (deduped_kernel_id_map[kernel_id] == kernel_id)
            {
                auto kernel_emitter = block_executor_program.block_kernels[kernel_id];
                NNFUSION_CHECK_NOT_NULLPTR(kernel_emitter);
                lu << kernel_emitter->emit_block_kernel()->get_code();
            }
        }
    }
    else
    {
        for (auto kernel_emitter : block_executor_program.block_kernels)
        {
            NNFUSION_CHECK_NOT_NULLPTR(kernel_emitter);
            lu << kernel_emitter->emit_block_kernel()->get_code();
        }
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_block_executor_instruction_execute_block(
    std::shared_ptr<BlockExecutorInstructionExecuteBlock> be_ins_execute_block)
{
    NNFUSION_CHECK_NOT_NULLPTR(be_ins_execute_block);

    LanguageUnit_p _lu(new LanguageUnit());
    auto& lu = *_lu;

    auto kernel_id = be_ins_execute_block->kernel_id;
    auto deduped_kernel_id = kernel_id;
    if (this->is_dedupe_block_kernels == true)
    {
        deduped_kernel_id = deduped_kernel_id_map[kernel_id];
    }
    auto kernel_emitter = block_executor_program.block_kernels[kernel_id];
    auto deduped_kernel_emitter = block_executor_program.block_kernels[deduped_kernel_id];

    std::vector<std::string> params;
    for (size_t i = 0; i < kernel_emitter->m_context->inputs.size(); i++)
    {
        std::stringstream ss;
        ss << all_args[kernel_emitter->m_context->inputs[i]->get_name()];
        params.push_back(ss.str());
    }
    for (size_t i = 0; i < kernel_emitter->m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << all_args[kernel_emitter->m_context->outputs[i]->get_name()];
        params.push_back(ss.str());
    }
    params.push_back("threadIdx.x");
    params.push_back(std::to_string(be_ins_execute_block->kernel_block_id));

    if (this->is_shared_buffer == false)
    {
        params.push_back("NULL");
    }
    else
    {
        params.push_back("shared_buffer");
    }

    lu << deduped_kernel_emitter->get_function_name() << "_block_kernel"
       << "(" << join(params, ", ") << ");"
       << "\n";

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_block_executor_instruction_step_to(
    std::shared_ptr<BlockExecutorInstructionStepTo> be_ins_step_to)
{
    NNFUSION_CHECK_NOT_NULLPTR(be_ins_step_to);

    LanguageUnit_p _lu(new LanguageUnit());
    auto& lu = *_lu;

    lu << "BlockFusion_step_to_device_function(be_state_buffer, "
       << std::to_string(be_ins_step_to->be_id) << ", " << std::to_string(be_ins_step_to->step_id)
       << ");\n";

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_block_executor_instruction_wait_for(
    std::shared_ptr<BlockExecutorInstructionWaitFor> be_ins_wait_for)
{
    NNFUSION_CHECK_NOT_NULLPTR(be_ins_wait_for);

    LanguageUnit_p _lu(new LanguageUnit());
    auto& lu = *_lu;

    bool flag_range_wait_for = true;
    for (int i = 1; i < be_ins_wait_for->bes_predecessor.size(); i++)
    {
        if (be_ins_wait_for->bes_predecessor[i] - be_ins_wait_for->bes_predecessor[i - 1] != 1)
        {
            flag_range_wait_for = false;
            break;
        }
    }
    if (this->get_block_dim().x < be_ins_wait_for->bes_predecessor.size())
    {
        flag_range_wait_for = false;
    }

    if (flag_range_wait_for)
    {
        lu.block_begin();
        lu << "if (threadIdx.x < " << be_ins_wait_for->bes_predecessor.size() << ")\n";
        lu.block_begin();
        lu << "while (be_state_buffer[" << be_ins_wait_for->bes_predecessor[0]
           << " + threadIdx.x] < " << be_ins_wait_for->step_id << ") {}\n";
        lu.block_end();
        lu << "__syncthreads();\n";
        lu.block_end();
    }
    else
    {
        lu.block_begin();
        lu << "if (threadIdx.x == 0)\n";
        lu.block_begin();
        for (auto be_predecessor : be_ins_wait_for->bes_predecessor)
        {
            lu << "while (be_state_buffer[" << be_predecessor << "] < " << be_ins_wait_for->step_id
               << ") {}\n";
        }
        lu.block_end();
        lu << "__syncthreads();\n";
        lu.block_end();
    }

    return _lu;
}

LanguageUnit_p
    BlockFusionCudaCodegen::emit_block_executor_instruction(BEInstruction_p be_instruction)
{
    NNFUSION_CHECK_NOT_NULLPTR(be_instruction);

    LanguageUnit_p _lu(new LanguageUnit("be_instruction"));
    LanguageUnit& lu = *_lu;

    if (auto ins_execute_block =
            std::dynamic_pointer_cast<BlockExecutorInstructionExecuteBlock>(be_instruction))
    {
        lu << emit_block_executor_instruction_execute_block(ins_execute_block)->get_code();
    }
    else if (auto ins_step_to =
                 std::dynamic_pointer_cast<BlockExecutorInstructionStepTo>(be_instruction))
    {
        lu << emit_block_executor_instruction_step_to(ins_step_to)->get_code();
    }
    else if (auto ins_wait_for =
                 std::dynamic_pointer_cast<BlockExecutorInstructionWaitFor>(be_instruction))
    {
        lu << emit_block_executor_instruction_wait_for(ins_wait_for)->get_code();
    }
    else
    {
        NNFUSION_CHECK_FAIL()
            << "BlockFusionCudaCodegen: do not support this BlockExecutorInstruction";
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_block_executor(int be_id)
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_be_" + std::to_string(be_id)));
    LanguageUnit& lu = *_lu;

    auto be = block_executor_program.block_executor_instructions[be_id];

    for (auto be_instruction : be)
    {
        lu << emit_block_executor_instruction(be_instruction)->get_code();
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_range_block_executor(int be_st_id, int be_ed_id)
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_be_" + std::to_string(be_st_id) +
                                        "_to_be_" + std::to_string(be_ed_id)));
    LanguageUnit& lu = *_lu;

    if (be_st_id == be_ed_id)
    {
        return emit_block_executor(be_st_id);
    }

    auto be_st = block_executor_program.block_executor_instructions[be_st_id];
    auto be_ed = block_executor_program.block_executor_instructions[be_ed_id];

    NNFUSION_CHECK(be_st.size() == be_ed.size());

    for (int ins_idx = 0; ins_idx < be_st.size(); ins_idx++)
    {
        auto ins_st = be_st[ins_idx];
        auto ins_ed = be_ed[ins_idx];

        NNFUSION_CHECK_NOT_NULLPTR(ins_st);
        NNFUSION_CHECK_NOT_NULLPTR(ins_ed);

        auto ins_st_execute_block =
            std::dynamic_pointer_cast<BlockExecutorInstructionExecuteBlock>(ins_st);
        auto ins_ed_execute_block =
            std::dynamic_pointer_cast<BlockExecutorInstructionExecuteBlock>(ins_ed);
        auto ins_st_step_to = std::dynamic_pointer_cast<BlockExecutorInstructionStepTo>(ins_st);
        auto ins_ed_step_to = std::dynamic_pointer_cast<BlockExecutorInstructionStepTo>(ins_ed);
        auto ins_st_wait_for = std::dynamic_pointer_cast<BlockExecutorInstructionWaitFor>(ins_st);
        auto ins_ed_wait_for = std::dynamic_pointer_cast<BlockExecutorInstructionWaitFor>(ins_ed);

        if (ins_st_execute_block != nullptr && ins_ed_execute_block != nullptr)
        {
            NNFUSION_CHECK(ins_st_execute_block->kernel_id == ins_ed_execute_block->kernel_id);
            NNFUSION_CHECK((ins_ed_execute_block->be_id - ins_st_execute_block->be_id) ==
                           (be_ed_id - be_st_id));
            NNFUSION_CHECK((ins_ed_execute_block->kernel_block_id -
                            ins_st_execute_block->kernel_block_id) == (be_ed_id - be_st_id));

            auto kernel_id = ins_st_execute_block->kernel_id;
            auto deduped_kernel_id = kernel_id;
            if (this->is_dedupe_block_kernels == true)
            {
                deduped_kernel_id = deduped_kernel_id_map[kernel_id];
            }
            auto kernel_emitter = block_executor_program.block_kernels[kernel_id];
            auto deduped_kernel_emitter = block_executor_program.block_kernels[deduped_kernel_id];

            std::vector<std::string> params;
            for (size_t i = 0; i < kernel_emitter->m_context->inputs.size(); i++)
            {
                std::stringstream ss;
                ss << all_args[kernel_emitter->m_context->inputs[i]->get_name()];
                params.push_back(ss.str());
            }
            for (size_t i = 0; i < kernel_emitter->m_context->outputs.size(); i++)
            {
                stringstream ss;
                ss << all_args[kernel_emitter->m_context->outputs[i]->get_name()];
                params.push_back(ss.str());
            }
            params.push_back("threadIdx.x");
            params.push_back("blockIdx.x - " + std::to_string(be_st_id) + " + " +
                             std::to_string(ins_st_execute_block->kernel_block_id));

            if (this->is_shared_buffer == false)
            {
                params.push_back("NULL");
            }
            else
            {
                params.push_back("shared_buffer");
            }

            lu << deduped_kernel_emitter->get_function_name() << "_block_kernel"
               << "(" << join(params, ", ") << ");"
               << "\n";
        }
        else if (ins_st_step_to != nullptr && ins_ed_step_to != nullptr)
        {
            NNFUSION_CHECK(ins_st_step_to->step_id == ins_ed_step_to->step_id);
            NNFUSION_CHECK((ins_ed_step_to->be_id - ins_st_step_to->be_id) ==
                           (be_ed_id - be_st_id));

            lu << "BlockFusion_step_to_device_function(be_state_buffer, "
               << ("blockIdx.x - " + std::to_string(be_st_id) + " + " +
                   std::to_string(ins_st_step_to->be_id))
               << ", " << std::to_string(ins_st_step_to->step_id) << ");\n";
        }
        else if (ins_st_wait_for != nullptr && ins_ed_wait_for != nullptr)
        {
            NNFUSION_CHECK(ins_st_wait_for->step_id == ins_ed_wait_for->step_id);

            lu << emit_block_executor_instruction_wait_for(ins_st_wait_for)->get_code();

            // bool flag_range_wait_for = true;
            // for (int i = 1; i < ins_st_wait_for->bes_predecessor.size(); i++)
            // {
            //     if (ins_st_wait_for->bes_predecessor[i] - ins_st_wait_for->bes_predecessor[i - 1] !=
            //         1)
            //     {
            //         flag_range_wait_for = false;
            //         break;
            //     }
            // }
            // if (this->get_block_dim().x < ins_st_wait_for->bes_predecessor.size())
            // {
            //     flag_range_wait_for = false;
            // }

            // if (flag_range_wait_for)
            // {
            //     lu.block_begin();
            //     // lu << "__syncthreads();\n";
            //     lu << "if (threadIdx.x < " << ins_st_wait_for->bes_predecessor.size() << ")\n";
            //     lu.block_begin();
            //     lu << "while (be_state_buffer[" << ins_st_wait_for->bes_predecessor[0]
            //        << " + threadIdx.x] < " << ins_st_wait_for->step_id << ") {}\n";
            //     lu.block_end();
            //     lu << "__syncthreads();\n";
            //     lu.block_end();
            // }
            // else
            // {
            //     lu.block_begin();
            //     // lu << "__syncthreads();\n";
            //     lu << "if (threadIdx.x == 0)\n";
            //     lu.block_begin();
            //     for (auto be_predecessor : ins_st_wait_for->bes_predecessor)
            //     {
            //         lu << "while (be_state_buffer[" << be_predecessor << "] < "
            //            << ins_st_wait_for->step_id << ") {}\n";
            //     }
            //     lu.block_end();
            //     lu << "__syncthreads();\n";
            //     lu.block_end();
            // }
        }
        else
        {
            NNFUSION_CHECK_FAIL() << "BlockFusionCudaCodegen: do not support this "
                                     "BlockExecutorInstruction or range check failed";
        }
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_sig"));
    auto& lu = *_lu;

    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
        ss << "input" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->tensors.size(); i++)
    {
        stringstream ss;
        ss << m_context->tensors[i]->get_element_type().c_type_string() << "* ";
        ss << "temp" << i;
        params.push_back(ss.str());
    }

    // be_state_buffer for group_sync
    if (this->is_group_sync)
    {
        params[params.size() - 1] = "volatile int* be_state_buffer";
    }

    lu << "extern \"C\" __global__  void "
       << "(" << join(params, ", ") << ")";

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_alloc_shared()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_alloc_shared"));
    LanguageUnit& lu = *_lu;

    size_t kernel_shared_size = 0;

    for (auto kernel : m_context->kernels)
    {
        auto block_kernel = std::dynamic_pointer_cast<BlockCudaEmitter>(kernel);
        NNFUSION_CHECK_NOT_NULLPTR(block_kernel);
        kernel_shared_size = std::max(kernel_shared_size, block_kernel->get_shared_memory_size());
    }

    // avoid allocate shared_memory when no kernels use shared_memory
    if (kernel_shared_size == 0)
    {
        this->is_shared_buffer = false;
    }
    else
    {
        this->is_shared_buffer = true;
        lu << "__shared__ char shared_buffer[" << std::to_string(kernel_shared_size) << "];"
           << "\n";
    }

    // alloc shared for block sync

    lu << "\n";

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_function_body_range_branch()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    LanguageUnit& lu = *_lu;

    lu << emit_alloc_shared()->get_code();

    if (this->is_group_sync)
    {
        // lu << "BlockFusion_group_sync_init_device_function(be_state_buffer);\n";
    }

    std::vector<std::pair<int, int>> ranges;
    int be_st = 0;
    int be_ed = 0;
    while (be_st < block_executor_program.block_executor_instructions.size())
    {
        if (block_executor_program.block_executor_instructions[be_st].size() == 0)
        {
            be_st += 1;
            continue;
        }
        be_ed = be_st;
        bool flag_be_sequential = true;
        for (be_ed = be_st + 1; be_ed < block_executor_program.block_executor_instructions.size();
             be_ed++)
        {
            auto& bes_a = block_executor_program.block_executor_instructions[be_ed - 1];
            auto& bes_b = block_executor_program.block_executor_instructions[be_ed];
            if (bes_a.size() != bes_b.size())
            {
                flag_be_sequential = false;
                break;
            }
            for (int i = 0; i < bes_a.size(); i++)
            {
                if (!check_instruction_sequential(bes_a[i], bes_b[i]))
                {
                    flag_be_sequential = false;
                    break;
                }
            }

            if (!flag_be_sequential)
            {
                break;
            }
        }
        ranges.push_back(make_pair(be_st, be_ed - 1));
        be_st = be_ed;
    }

    bool flag_first_branch = true;
    for (auto range : ranges)
    {
        if (range.first == range.second)
        {
            if (flag_first_branch)
            {
                lu << "if (blockIdx.x == " << std::to_string(range.first) << ")\n";
                flag_first_branch = false;
            }
            else
            {
                lu << "else if (blockIdx.x == " << std::to_string(range.first) << ")\n";
            }
            lu.block_begin();
            lu << emit_range_block_executor(range.first, range.second)->get_code();
            lu.block_end();
        }
        else
        {
            if (flag_first_branch)
            {
                lu << "if ((int)blockIdx.x >= " << std::to_string(range.first)
                   << " && (int)blockIdx.x <= " << std::to_string(range.second) << ")\n";
                flag_first_branch = false;
            }
            else
            {
                lu << "else if ((int)blockIdx.x >= " << std::to_string(range.first)
                   << " && (int)blockIdx.x <= " << std::to_string(range.second) << ")\n";
            }
            lu.block_begin();
            lu << emit_range_block_executor(range.first, range.second)->get_code();
            lu.block_end();
        }
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_range_branch(int kernel_id,
                                                         int be_st,
                                                         int be_ed,
                                                         bool flag_first_branch)
{
    // TODO(v-lima): remove this function and use emit_range_block_executor
    LanguageUnit_p _lu(new LanguageUnit());
    auto& lu = *_lu;

    auto deduped_kernel_id = kernel_id;
    if (this->is_dedupe_block_kernels == true)
    {
        deduped_kernel_id = deduped_kernel_id_map[kernel_id];
    }
    auto kernel_emitter = block_executor_program.block_kernels[kernel_id];
    auto deduped_kernel_emitter = block_executor_program.block_kernels[deduped_kernel_id];

    std::vector<std::string> params;
    for (size_t i = 0; i < kernel_emitter->m_context->inputs.size(); i++)
    {
        std::stringstream ss;
        ss << all_args[kernel_emitter->m_context->inputs[i]->get_name()];
        params.push_back(ss.str());
    }
    for (size_t i = 0; i < kernel_emitter->m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << all_args[kernel_emitter->m_context->outputs[i]->get_name()];
        params.push_back(ss.str());
    }
    params.push_back("threadIdx.x");
    params.push_back("blockIdx.x - " + std::to_string(be_st));

    if (this->is_shared_buffer == false)
    {
        params.push_back("NULL");
    }
    else
    {
        params.push_back("shared_buffer");
    }

    if (flag_first_branch)
    {
        lu << "if ((int)blockIdx.x >= " << std::to_string(be_st)
           << " && (int)blockIdx.x <= " << std::to_string(be_ed) << ")\n";
    }
    else
    {
        lu << "else if ((int)blockIdx.x >= " << std::to_string(be_st)
           << " && (int)blockIdx.x <= " << std::to_string(be_ed) << ")\n";
    }
    lu.block_begin();
    lu << deduped_kernel_emitter->get_function_name() << "_block_kernel"
       << "(" << join(params, ", ") << ");"
       << "\n";
    lu.block_end();

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_function_body_simple_range_branch()
{
    // TODO(v-lima): re-construct, use emit_range_block_executor
    for (int be_id = 0; be_id < block_executor_program.block_executor_instructions.size(); be_id++)
    {
        if (block_executor_program.block_executor_instructions[be_id].size() > 1)
        {
            return nullptr;
        }
        if (block_executor_program.block_executor_instructions[be_id].size() > 0)
        {
            auto be_instruction = block_executor_program.block_executor_instructions[be_id][0];
            auto ins_execute_block =
                std::dynamic_pointer_cast<BlockExecutorInstructionExecuteBlock>(be_instruction);
            if (ins_execute_block == nullptr)
            {
                return nullptr;
            }
        }
    }

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    LanguageUnit& lu = *_lu;

    lu << emit_alloc_shared()->get_code();

    if (this->is_group_sync)
    {
        // lu << "BlockFusion_group_sync_init_device_function(be_state_buffer);\n";
    }

    bool flag_first_be_id = true;
    int kernel_block_schedule_checker = 0;
    int emit_be_st = 0;
    int emit_be_ed = 0;
    int emit_kernel_id = -1;
    int current_kernel_id = -1;
    int current_kernel_block_id = 0;
    for (int be_id = 0; be_id < block_executor_program.block_executor_instructions.size(); be_id++)
    {
        if (block_executor_program.block_executor_instructions[be_id].size() == 0)
        {
            current_kernel_id = -1;
            if (emit_kernel_id != -1)
            {
                emit_be_ed = be_id - 1;
                lu << emit_range_branch(emit_kernel_id, emit_be_st, emit_be_ed, flag_first_be_id)
                          ->get_code();
                flag_first_be_id = false;
            }
            emit_kernel_id = -1;
            emit_be_st = emit_be_ed = be_id;
            kernel_block_schedule_checker = 0;
        }
        else
        {
            auto be_instruction = block_executor_program.block_executor_instructions[be_id][0];
            auto ins_execute_block =
                std::dynamic_pointer_cast<BlockExecutorInstructionExecuteBlock>(be_instruction);
            current_kernel_id = ins_execute_block->kernel_id;
            current_kernel_block_id = ins_execute_block->kernel_block_id;
            if (emit_kernel_id == current_kernel_id)
            {
                emit_be_ed = be_id;
                // check schedule policy
                if (current_kernel_block_id - be_id != kernel_block_schedule_checker)
                {
                    return nullptr;
                }
            }
            else
            {
                if (emit_kernel_id != -1)
                {
                    emit_be_ed = be_id - 1;
                    lu << emit_range_branch(
                              emit_kernel_id, emit_be_st, emit_be_ed, flag_first_be_id)
                              ->get_code();
                    flag_first_be_id = false;
                }
                emit_kernel_id = current_kernel_id;
                emit_be_st = be_id;
                emit_be_ed = be_id;
                kernel_block_schedule_checker = current_kernel_block_id - be_id;
            }
        }
    }
    if (emit_kernel_id != -1)
    {
        lu << emit_range_branch(emit_kernel_id, emit_be_st, emit_be_ed, flag_first_be_id)
                  ->get_code();
        flag_first_be_id = false;
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_function_body_default()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    LanguageUnit& lu = *_lu;

    lu << emit_alloc_shared()->get_code();

    if (this->is_group_sync)
    {
        // lu << "BlockFusion_group_sync_init_device_function(be_state_buffer);\n";
    }

    bool flag_first_be_id = true;
    for (int be_id = 0; be_id < block_executor_program.block_executor_instructions.size(); be_id++)
    {
        // skip empty BEs when there are no kernels in some BEs.
        if (block_executor_program.block_executor_instructions[be_id].size() > 0)
        {
            if (flag_first_be_id)
            {
                lu << "if (blockIdx.x == " << std::to_string(be_id) << ")\n";
                flag_first_be_id = false;
            }
            else
            {
                lu << "else if (blockIdx.x == " << std::to_string(be_id) << ")\n";
            }
            lu.block_begin();
            lu << emit_block_executor(be_id)->get_code();
            lu.block_end();
        }
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_function_body()
{
    NNFUSION_LOG(DEBUG) << "BlockFusionCudaCodegen: codegen optimization level: "
                        << this->codegen_opt_level;
    LanguageUnit_p _lu;
    if (this->codegen_opt_level == 1)
    {
        _lu = emit_function_body_simple_range_branch();
    }
    else if (this->codegen_opt_level == 2)
    {
        _lu = emit_function_body_range_branch();
    }

    if (_lu == nullptr)
    {
        NNFUSION_LOG(DEBUG)
            << "BlockFusionCudaCodegen: codegen optimization failed, fallback to default style";
        _lu = emit_function_body_default();
    }
    NNFUSION_CHECK_NOT_NULLPTR(_lu) << "BlockFusionCudaCodegen::emit_function_body failed";

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::stdio);

    // keep each kernel's dependency
    for (auto kernel : m_context->kernels)
    {
        auto kernel_dep = kernel->get_or_emit_source()->dep_unit;
        for (auto& it : kernel_dep->local_symbol)
        {
            _lu->require(it.second);
        }
    }

    if (this->is_group_sync)
    {
        _lu->require(emit_step_to_device_function());
        // _lu->require(emit_wait_for_device_function());
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_function_name()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_function_name"));
    LanguageUnit& lu = *_lu;

    std::vector<std::string> names;
    for (auto kernel : m_context->kernels)
    {
        names.push_back(kernel->m_context->gnode->get_op_type());
    }

    lu << "BlockFusionKernel_" << join(m_context->dtypes, "_") << "_" << m_kernel_type << "_"
       << join(names, "_") << "_" << BlockFusionCudaCodegen::unique_func_id++; //<< custom_tag;

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_comments()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_comments"));
    LanguageUnit& lu = *_lu;

    lu << "// Node name:\t BlockFusion"
       << "\n";
    //lu << "// Description:\t" << m_context->node->description() << "\n";
    lu << "// Input:\n";
    for (auto in : m_context->inputs)
    {
        lu << "//\t- name: " << in->get_name();
        lu << "\ttype: " << in->get_element_type().c_type_string();
        lu << "\tshape: " << in->get_shape();
        lu << "\n";
    }

    lu << "// Output:\n";
    for (auto out : m_context->outputs)
    {
        lu << "//\t- name: " << out->get_name();
        lu << "\ttype: " << out->get_element_type().c_type_string();
        lu << "\tshape: " << out->get_shape();
        lu << "\n";
    }

    if (!m_context->tensors.empty())
    {
        lu << "// Other tensors in use:\n";
        for (auto persist : m_context->tensors)
        {
            lu << "//\t- name: " << persist->get_name();
            lu << "\ttype: " << persist->get_element_type().c_type_string();
            lu << "\tshape: " << persist->get_shape();
            lu << "\n";
        }
    }

    lu << "// Fused functions:\n";
    for (auto kernel : m_context->kernels)
    {
        lu << "// " << kernel->get_or_emit_source()->name_unit->get_code()
           << kernel->get_or_emit_source()->call_unit->get_code();
    }

    if (is_dedupe_block_kernels == true)
    {
        lu << "// Deduped function map: <src_function_name : deduped_function_name>\n";
        for (int kernel_id = 0; kernel_id < block_executor_program.block_kernels.size();
             kernel_id++)
        {
            if (kernel_id != deduped_kernel_id_map[kernel_id])
            {
                lu << "// " << block_executor_program.block_kernels[kernel_id]->get_function_name()
                   << " : "
                   << block_executor_program.block_kernels[deduped_kernel_id_map[kernel_id]]
                          ->get_function_name()
                   << "\n";
            }
        }
    }

    // emit block kernel functions here
    lu << emit_block_kernel_functions()->get_code();

    return _lu;
}

void BlockFusionCudaCodegen::set_launch_config()
{
    int grids, blocks, bound;
    compute_launch_config(grids, blocks, bound);

    m_gridDim = dim3(grids, 1, 1);
    m_blockDim = dim3(blocks, 1, 1);
}

void BlockFusionCudaCodegen::compute_launch_config(int& grids, int& blocks, int& bound)
{
    grids = block_executor_program.num_bes;
    // launch less thread_blocks when there are no kernels in some BEs.
    for (int be_id = block_executor_program.num_bes - 1; be_id >= 0; be_id--)
    {
        if (block_executor_program.block_executor_instructions[be_id].size() == 0)
        {
            grids = be_id;
        }
        else
        {
            break;
        }
    }

    blocks = 0;
    for (auto kernel : m_context->kernels)
    {
        auto block_kernel = std::dynamic_pointer_cast<BlockCudaEmitter>(kernel);
        NNFUSION_CHECK_NOT_NULLPTR(block_kernel);
        dim3 kernel_block_dim = block_kernel->get_block_dim();
        blocks = std::max(blocks, kernel_block_dim.x * kernel_block_dim.y * kernel_block_dim.z);
    }
}

LanguageUnit_p BlockFusionCudaCodegen::emit_function_call()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
    auto& lu = *_lu;
    vector<string> names;
    set_launch_config();

    string stream_name = "0";
    auto gnode = m_context->gnode;
    if (gnode != nullptr)
    {
        NNFUSION_CHECK_NOT_NULLPTR(gnode);
        if ((*gnode)["Async_info"].is_valid())
        {
            auto& async_info = (*gnode)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
            if (async_info.execution_stream != nullptr)
                stream_name = async_info.execution_stream->get_name();
        }
    }

    names.insert(names.end(), m_context->input_names.begin(), m_context->input_names.end());
    names.insert(names.end(), m_context->output_names.begin(), m_context->output_names.end());
    names.insert(names.end(), m_context->tensor_names.begin(), m_context->tensor_names.end());
    // for group_sync
    // if (this->is_group_sync)
    // {
    //     names.push_back(this->m_context->tensors[0]->get_name());
    // }

    lu << "<<<dim3(" << m_gridDim.x << ", " << m_gridDim.y << ", " << m_gridDim.z << "), dim3("
       << m_blockDim.x << ", " << m_blockDim.y << ", " << m_blockDim.z << "), 0, " << stream_name
       << ">>>(" << join(names, ", ") << ");\n";

    return _lu;
}

bool BlockFusionCudaCodegen::check_instruction_sequential(BEInstruction_p ins_a,
                                                          BEInstruction_p ins_b)
{
    NNFUSION_CHECK_NOT_NULLPTR(ins_a);
    NNFUSION_CHECK_NOT_NULLPTR(ins_b);

    auto ins_a_execute_block =
        std::dynamic_pointer_cast<BlockExecutorInstructionExecuteBlock>(ins_a);
    auto ins_b_execute_block =
        std::dynamic_pointer_cast<BlockExecutorInstructionExecuteBlock>(ins_b);
    auto ins_a_step_to = std::dynamic_pointer_cast<BlockExecutorInstructionStepTo>(ins_a);
    auto ins_b_step_to = std::dynamic_pointer_cast<BlockExecutorInstructionStepTo>(ins_b);
    auto ins_a_wait_for = std::dynamic_pointer_cast<BlockExecutorInstructionWaitFor>(ins_a);
    auto ins_b_wait_for = std::dynamic_pointer_cast<BlockExecutorInstructionWaitFor>(ins_b);

    if (ins_a_execute_block != nullptr && ins_b_execute_block != nullptr)
    {
        if (ins_a_execute_block->kernel_id != ins_b_execute_block->kernel_id)
        {
            return false;
        }
        if ((ins_b_execute_block->be_id - ins_a_execute_block->be_id) != 1)
        {
            return false;
        }
        if ((ins_b_execute_block->kernel_block_id - ins_a_execute_block->kernel_block_id) != 1)
        {
            return false;
        }
        return true;
    }
    else if (ins_a_step_to != nullptr && ins_b_step_to != nullptr)
    {
        if (ins_a_step_to->step_id != ins_b_step_to->step_id)
        {
            return false;
        }
        if ((ins_b_step_to->be_id - ins_a_step_to->be_id) != 1)
        {
            return false;
        }
        return true;
    }
    else if (ins_a_wait_for != nullptr && ins_b_wait_for != nullptr)
    {
        if (ins_a_wait_for->step_id != ins_b_wait_for->step_id)
        {
            return false;
        }
        if (ins_a_wait_for->bes_predecessor.size() != ins_b_wait_for->bes_predecessor.size())
        {
            return false;
        }
        for (int i = 0; i < ins_a_wait_for->bes_predecessor.size(); i++)
        {
            if (ins_a_wait_for->bes_predecessor[i] != ins_b_wait_for->bes_predecessor[i])
            {
                return false;
            }
        }
        return true;
    }

    return false;
}