// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "base_codegen_pass.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::codegen;

DECLARE_bool(fkernels_as_files);
DECLARE_int64(fkernels_files_number);
DECLARE_bool(fcustomized_mem_imp);
DECLARE_string(fantares_perf_file);
DECLARE_bool(ffunction_codegen);
bool BaseCodegenPass::run(std::shared_ptr<InterpreterContext> ctx,
                          std::shared_ptr<TranslationUnit> tu)
{
    initialize(ctx, tu);
    NNFUSION_CHECK(collect_mem(ctx, tu));
    NNFUSION_CHECK(collect_stream(ctx, tu));
    NNFUSION_CHECK(collect_funcs(ctx, tu));
    NNFUSION_CHECK(modify_codegen());

    // codegen
    projgen->codegen();
    NNFUSION_CHECK(after_projgen());
    NNFUSION_LOG(INFO) << "Codegen for " << get_device_str(device_type()) << " done.";
    exit(0);

    return true;
}

void BaseCodegenPass::initialize(std::shared_ptr<InterpreterContext> ctx,
                                 std::shared_ptr<TranslationUnit> tu)
{
    // setup lup_codegen execution info
    projgen->lup_codegen->pwd = m_codegen_folder;
    projgen->lup_codegen->write_to = "nnfusion_rt.h";

    return;
}

nnfusion::codegen::CodegenFuncCallsUnit_p
    BaseCodegenPass::get_kernel_func_calls(const string& calls_symbol,
                                           CodegenMainBlockUnit_p main_block)
{
    std::string search_str = calls_symbol;
    if (main_block)
        search_str += "_" + main_block->symbol;
    if (kernel_func_calls.find(search_str) != kernel_func_calls.end())
        return kernel_func_calls[search_str];

    CodegenFuncCallsUnit_p lup_func_calls = std::make_shared<CodegenFuncCallsUnit>(calls_symbol);
    // add to main_block
    if (main_block)
        main_block->unit_vec.push_back(lup_func_calls);
    kernel_func_calls[search_str] = lup_func_calls;
    return kernel_func_calls[search_str];
}

void BaseCodegenPass::separate_func_defs_files(int file_number, const string& kernel_folder)
{
    if (kernel_folder != m_kernel_folder)
        change_kernel_folder(kernel_folder);

    if (file_number <= 0)
    {
        for (auto it : kernel_func_defs)
        {
            LanguageUnit_p func_def = it.second.second;
            if (func_def->pwd.empty())
                func_def->pwd = kernel_folder;
            string fname = func_def->symbol;
            if (fname.length() > 128)
            {
                size_t hashcode = std::hash<std::string>{}(fname);
                fname = "compressed_src_" + std::to_string(hashcode);
            }
            if (func_def->write_to.empty())
                func_def->write_to = fname + m_kernel_suffix;
        }
    }
    else
    {
        int i = 0;
        for (auto it : kernel_func_defs)
        {
            int file_idx = i % file_number;
            LanguageUnit_p func_def = it.second.second;
            func_def->pwd = kernel_folder;
            func_def->write_to = "kernel_func_def_" + to_string(file_idx) + m_kernel_suffix;
            i++;
        }
    }
}

void BaseCodegenPass::add_init_and_exit_pair(LanguageUnit_p lup_in_init, LanguageUnit_p lup_in_exit)
{
    //add to init
    projgen->lup_init->unit_vec.push_back(lup_in_init);
    //add to exit
    projgen->lup_exit->unit_vec.push_front(lup_in_exit);

    return;
}

bool BaseCodegenPass::collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu)
{
    if (!tu)
        return false;
    auto lup_func_calls = get_kernel_func_calls("func_calls", projgen->lup_exec);

    auto& prog = tu->program;
    // collect code
    for (auto iterator : prog)
    {
        for (auto ins : *iterator)
        {
            KernelEmitter::Pointer kernel;
            kernel = ins->getKernel();
            if (!kernel || !kernel->get_or_emit_source())
            {
                return false;
            }

            // process kernel code
            FunctionUnit_p fu = kernel->get_or_emit_source(true);
            string call_str = fu->call_unit->get_code();
            string body_str = fu->body_unit->get_code();
            if (!body_str.empty())
            {
                if (kernel_func_defs.find(body_str) == kernel_func_defs.end())
                {
                    auto kernel_func_def = fu->body_unit;
                    for (auto& it : fu->dep_unit->local_symbol)
                    {
                        kernel_func_def->require(it.second);
                    }
                    kernel_func_defs[body_str] = make_pair(call_str, kernel_func_def);
                }
                else
                {
                    call_str = kernel_func_defs[body_str].first;
                }
            }
            LanguageUnit_p kernel_func_call = fu->call_unit;
            lup_func_calls->unit_vec.push_back(kernel_func_call);
            lup_func_calls->require(kernel_func_defs[body_str].second);
            if (FLAGS_fkernels_as_files &&
                kernel_func_defs[body_str].second->extern_decl_unit != nullptr)
                lup_func_calls->require(kernel_func_defs[body_str].second->extern_decl_unit);
        }
    }

    if (FLAGS_fkernels_as_files)
        separate_func_defs_files(FLAGS_fkernels_files_number, m_kernel_folder);

    return true;
}

bool BaseCodegenPass::collect_mem(std::shared_ptr<InterpreterContext> ctx,
                                  std::shared_ptr<TranslationUnit> tu)
{
    if (!tu)
        return false;
    auto mem_pair = create_init_and_exit_pair<LanguageUnitwithVec, LanguageUnitwithVec>("MEM_ALLOC",
                                                                                        "MEM_FREE");
    auto lup_mem_alloc = mem_pair.first;
    auto lup_mem_free = mem_pair.second;
    auto& allocator_list = tu->memory_allocator_factory->get_allocator_list();

    size_t total_alloc = 0;
    for (const auto& allocator : allocator_list)
    {
        total_alloc += allocator.second->max_allocated();
    }
    LanguageUnit_p total = std::make_shared<LanguageUnit>(
        "total_memory", "// total memory:" + to_string(total_alloc) + "\n");
    lup_mem_alloc->unit_vec.push_back(total);

    size_t offset = 0;
    for (const auto& allocator : allocator_list)
    {
        auto init = allocator.second->emit_memory_init();
        auto alloc = allocator.second->emit_memory_alloc();
        auto free = allocator.second->emit_memory_free();

        if (FLAGS_ffunction_codegen)
        {
            auto mempool_offset = allocator.second->emit_memory_pool_offset(offset);
            offset += allocator.second->max_allocated();
            lup_mem_alloc->unit_vec.push_back(mempool_offset);
        }
        lup_mem_alloc->unit_vec.push_back(alloc);
        lup_mem_alloc->require(init);
        lup_mem_free->unit_vec.push_back(free);
        lup_mem_free->require(init);
    }

    return true;
}

bool BaseCodegenPass::after_projgen()
{
    struct stat s;
    std::string constant_folder = get_current_dir_name() + std::string("/Constant");
    std::string para_info_json = get_current_dir_name() + std::string("/para_info.json");
    std::string antares_perf_path =
        get_current_dir_name() + std::string("/") + FLAGS_fantares_perf_file;
    if (stat(constant_folder.c_str(), &s) == 0)
    {
        std::string nnfusion_rt_const_folder = m_codegen_folder + std::string("Constant");
        std::string cmd;
        if (stat(nnfusion_rt_const_folder.c_str(), &s) == 0)
        {
            cmd = std::string("rm -rf ") + nnfusion_rt_const_folder;
            if (0 != system(cmd.c_str()))
            {
                throw nnfusion::errors::RuntimeError("Failed to remove constant folder.\n");
            }
        }
        cmd = std::string("mv ") + constant_folder + " " + m_codegen_folder;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError("Failed to move constant files.\n");
        }
    }
    if (stat(para_info_json.c_str(), &s) == 0)
    {
        std::string cmd = std::string("mv ") + para_info_json + " " + m_codegen_folder;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError("Failed to move para_info.json.\n");
        }
    }
    if (stat(antares_perf_path.c_str(), &s) == 0)
    {
        std::string cmd = std::string("mv ") + antares_perf_path + " " + m_codegen_folder;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError("Failed to move antares perf file.\n");
        }
    }

    return true;
}

std::pair<LanguageUnit_p, LanguageUnit_p>
    BaseCodegenPass::get_customized_mem_imp(nnfusion::ir::Instruction::Pointer ins)
{
    LanguageUnit_p lup_alloc(new LanguageUnit(ins->name() + "_alloc"));
    LanguageUnit_p lup_free(new LanguageUnit(ins->name() + "_free"));
    LanguageUnit_p lup_free_at_last(new LanguageUnit(ins->name() + "_free_at_last"));
    if (FLAGS_fcustomized_mem_imp)
    {
        if ((*ins)["MemoryInfo"].is_valid())
        {
            nnfusion::pass::MemoryInfo mem_info =
                (*ins)["MemoryInfo"].as<nnfusion::pass::MemoryInfo>();
            for (auto tensor : mem_info.alloc_new)
            {
                *lup_alloc << tensor->get_name() << " = MemoryAlloc(" << tensor->get_name() << ", "
                           << tensor->size() << ");\n";
                if (tensor->is_persistent() && free_at_last.find(tensor) == free_at_last.end())
                {
                    free_at_last.insert(tensor);
                    *lup_free_at_last << "MemoryFree(" << tensor->get_name() << ");\n";
                }
            }

            for (auto tensor : mem_info.alloc_ref)
            {
                auto root = tensor->get_root_tensor();
                NNFUSION_CHECK_NOT_NULLPTR(root);
                size_t offset = tensor->get_pool_offset() - root->get_pool_offset();
                *lup_alloc << tensor->get_name() << " = MemoryRef(" << tensor->get_name() << ", "
                           << root->get_name() << ", " << offset << ");\n";
                if (root->is_persistent() && free_at_last.find(root) == free_at_last.end())
                {
                    free_at_last.insert(root);
                    *lup_free_at_last << "MemoryFree(" << root->get_name() << ");\n";
                }
            }

            for (auto tensor : mem_info.free)
            {
                *lup_free << "MemoryFree(" << tensor->get_name() << ");\n";
            }
        }
    }

    if (!lup_free_at_last->get_code().empty())
        projgen->lup_exit->unit_vec.push_front(lup_free_at_last);
    return std::make_pair(lup_alloc, lup_free);
}

nnfusion::LanguageUnit_p BaseCodegenPass::codegen_mem_ref(KernelEmitter::Pointer kernel)
{
    if (!kernel || FLAGS_fcustomized_mem_imp)
        return nullptr;
    LanguageUnit_p _lu(new LanguageUnit(kernel->get_function_name() + "_mem_ref"));
    auto& lu = *_lu;
    bool empty = true;
    if (auto annotations = kernel->m_context->annotations)
    {
        for (auto oi_pair : annotations->get_in_place_oi_pairs())
        {
            if (oi_pair.force_inplace == true)
            {
                auto input = kernel->m_context->inputs[oi_pair.input];
                auto output = kernel->m_context->outputs[oi_pair.output];
                lu << output->get_name() << " = " << input->get_name() << ";\n";
                empty = false;
            }
        }
    }

    if (empty)
        return nullptr;
    return _lu;
}

LanguageUnit_p BaseCodegenPass::codegen_device_type()
{
    auto lu_devtype = make_shared<LanguageUnit>("device_type");
    *lu_devtype
        << "// 0: CUDA_GPU; 1: ROCM_GPU; 2: GENERIC_CPU; 3: HLSL; 4: GraphCore; 5: UNKNOWN\n";
    *lu_devtype << "int get_device_type()\n{\n";
    *lu_devtype << "    return " << device_type() << ";\n";
    *lu_devtype << "}\n";
    return lu_devtype;
}

LanguageUnit_p BaseCodegenPass::codegen_workspace_size(std::shared_ptr<TranslationUnit> tu)
{
    auto lu_workspace = make_shared<LanguageUnit>("workspace_size");

    auto& allocator_list = tu->memory_allocator_factory->get_allocator_list();

    size_t total_alloc = 0;
    for (const auto& allocator : allocator_list)
    {
        total_alloc += allocator.second->max_allocated();
    }

    *lu_workspace << "int get_workspace_size()\n{\n";
    *lu_workspace << "    return " << total_alloc << ";\n";
    *lu_workspace << "}\n";
    return lu_workspace;
}