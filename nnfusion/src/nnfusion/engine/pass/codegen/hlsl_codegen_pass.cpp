// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hlsl_codegen_pass.hpp"
#include "nnfusion/core/kernels/hlsl/hlsl_kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::codegen;

DECLARE_string(fdefault_device);

void HLSLCodegenPass::initialize(std::shared_ptr<InterpreterContext> ctx,
                                 std::shared_ptr<TranslationUnit> tu)
{
    // setup lup_codegen execution info
    projgen->lup_codegen->pwd = m_codegen_folder;
    projgen->lup_codegen->write_to = "nnfusion_rt.h";
    auto& copy_templates = projgen->lup_codegen->copy_templates;
    copy_templates.emplace_back("dxcompute/DxCompute.vcxproj", "./DxCompute.vcxproj");
    copy_templates.emplace_back("dxcompute/run_graph.cpp", "./run_graph.cpp");
    copy_templates.emplace_back("dxcompute/d3dx12_helper.h", "./d3dx12_helper.h");
    copy_templates.emplace_back("dxcompute/d3dx12_nnfusion.h", "./d3dx12_nnfusion.h");

    return;
}

bool HLSLCodegenPass::collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu)
{
    if (!tu)
        return false;

    auto lup_func_calls = get_kernel_func_calls("func_calls", projgen->lup_exec);

    auto& graph = tu->m_graph;
    auto& prog = tu->program;
    // collect code
    LanguageUnit_p begin = std::make_shared<LanguageUnit>("begin", "#if 1\n\n");
    lup_func_calls->unit_vec.push_back(begin);

    for (auto iterator : prog)
    {
        for (auto ins : *iterator)
        {
            KernelEmitter::Pointer kernel;
            kernel = ins->getKernel();
            if (!kernel || !kernel->get_or_emit_source())
            {
                UNHANDLED_CASE(ins->getGNode());
            }
            // process kernel code
            FunctionUnit_p fu = kernel->get_or_emit_source(true);
            string call_str = fu->call_unit->get_code();
            string body_str = fu->body_unit->get_code();
            if (!body_str.empty())
            {
                if (kernel_func_defs.find(body_str) == kernel_func_defs.end())
                {
                    std::string change_info = fu->body_unit->get_symbol();
                    auto kernel_func_def = fu->body_unit;
                    for (auto& it : fu->dep_unit->local_symbol)
                    {
                        kernel_func_def->require(it.second);
                    }

                    kernel_func_defs[body_str] = make_pair(change_info, kernel_func_def);
                }
                else
                {
                    int pos_left = call_str.find(", L\"");
                    int pos_right = call_str.find(".hlsl\"");
                    if (pos_left >= 0 && pos_right >= 0)
                        call_str.replace(pos_left + 4,
                                         pos_right - pos_left - 4,
                                         kernel_func_defs[body_str].first);
                }
            }
            LanguageUnit_p kernel_func_call =
                std::make_shared<LanguageUnit>(fu->call_unit->get_symbol(), call_str);
            lup_func_calls->unit_vec.push_back(kernel_func_call);
            if (kernel_func_defs.find(body_str) != kernel_func_defs.end())
                lup_func_calls->require(kernel_func_defs[body_str].second);
        }
    }
    LanguageUnit_p end = std::make_shared<LanguageUnit>("end", "#endif\n\n");
    lup_func_calls->unit_vec.push_back(end);

    LanguageUnit_p lup_cmd = make_shared<LanguageUnit>("COMMAND");
    auto& lu_cmd = *lup_cmd;
    {
        lu_cmd << R"(
  device.pCommandQueue->ExecuteCommandLists(preloadQueue.size(), preloadQueue.data());
  device.pCommandQueue->ExecuteCommandLists(cmdQueue.size(), cmdQueue.data());
  device.AwaitExecution();
)";
    }
    lup_func_calls->unit_vec.push_back(lup_cmd);

    LanguageUnit_p lup_print_res = make_shared<LanguageUnit>("PRINT_RES");
    auto& lu_print_res = *lup_print_res;
    {
        // Print Results
        for (auto& curr : graph->get_outputs()) // Print output nodes
        {
            if (tu->blacklist.count(curr))
                continue;
            lu_print_res << "op_" << curr->get_output_tensor_ptr(0)->get_name()
                         << ".PrintStageBuffer<" << curr->get_output_element_type(0).c_type_string()
                         << ">(device, \"ts_" << curr->get_output_tensor_ptr(0)->get_name()
                         << "\");\n";
        }
    }
    lup_func_calls->unit_vec.push_back(lup_print_res);

    separate_func_defs_files(-1, m_kernel_folder);

    return true;
}