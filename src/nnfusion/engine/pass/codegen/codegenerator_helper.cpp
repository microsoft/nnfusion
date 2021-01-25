// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "codegenerator_helper.hpp"

#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/engine/async_manager.hpp"

using namespace nnfusion;
using namespace nnfusion::codegenerator;
using namespace nnfusion::kernels;
using namespace nnfusion::async;

DECLARE_string(fhlsl_codegen_type);

LanguageUnit_p extern_function(LanguageUnit_p lu)
{
}

LanguageUnit_p extern_variable(LanguageUnit_p lu)
{
}

void FunctionFile::save_file()
{
    LanguageUnit def_re("require");
    def_re << boilerplate::MIT1->get_code();

    // Write Dependency
    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("header::") != string::npos)
            def_re << it.second->get_code();
    def_re << "\n";

    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("macro::") != string::npos)
            def_re << it.second->get_code();
    def_re << "\n";
    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("declaration::") != string::npos)
        {
            if (it.second->get_symbol() == "declaration::global_cublas_handle" ||
                it.second->get_symbol() == "declaration::global_cudnn_handle" ||
                it.second->get_symbol() == "declaration::num_SMs" ||
                it.second->get_symbol() == "declaration::allreduce_stream" ||
                it.second->get_symbol() == "declaration::applygradient_stream")
            {
                def_re << "extern ";
            }
            def_re << it.second->get_code();
        }
    def_re << "\n";

    string fname = this->get_symbol();
    if (fname.length() > 128)
    {
        size_t hashcode = std::hash<std::string>{}(fname);
        fname = "kernels/compressed_src_" + std::to_string(hashcode) + suffix_str;
    }
    else
        fname = "kernels/" + this->get_symbol() + suffix_str;

    ofstream src(fname);
    src << def_re.get_code();
    src << "\n";
    src << this->get_code();
    src.close();
}

void FunctionFile::merge_from(FunctionFile_p func)
{
    // Merge required symbols;
    for (auto& sym : func->local_symbol)
        require(sym.second);
    // Merge source code;
    (*this) << "\n" << func->get_code();
    // Merge symbol name;
    if (get_symbol() == "")
        change_symbol(func->get_symbol());
    else
        change_symbol(get_symbol() + "_" + func->get_symbol());
    extern_declare = extern_declare + "\n" + func->extern_declare;
}

FunctionFile::FunctionFile(string extern_declare, LanguageUnit_p file_context)
{
    // Get requirement
    this->clean_require();
    for (auto& sym : file_context->local_symbol)
        this->require(sym.second);
    // Get source code
    (*this) << file_context->get_code();
    change_symbol(file_context->get_symbol());
    this->extern_declare = extern_declare;
}

FunctionFile_p FunctionFile::convert_from(std::shared_ptr<nnfusion::kernels::KernelEmitter> kernel)
{
    FunctionUnit_p fu = kernel->get_or_emit_source();
    LanguageUnit_p lu = make_shared<LanguageUnit>();
    LanguageUnit& def = *lu;
    def.require(header::assert);
    def.require(header::stdexcept);
    def.require(header::sstream);
    def.require(header::cuda);
    def.require(header::cublas);
    def.require(header::cudnn);
    def.require(macro::CUDA_SAFE_CALL);
    def.require(macro::CUDNN_SAFE_CALL);
    def.require(macro::CUBLAS_SAFE_CALL);
    def.require(declaration::typedef_int);

    for (auto& sym : fu->dep_unit->local_symbol)
        def.require(sym.second);

    string body_unit = fu->body_unit->get_code();
    std::string sig = fu->get_specialized_signature();
    auto gnode = kernel->m_context->gnode;
    auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();

    // conv kernels in the same stream shares the same workspace_ptr
    if (gnode->get_op_type() == "Convolution")
    {
        std::string s_workspace =
            "workspace_ptr_" + to_string(async_info.execution_stream->get_stream_id());
        int pos = body_unit.find("workspace_ptr");
        while (pos >= 0)
        {
            body_unit.replace(pos, 13, s_workspace);
            pos = body_unit.find("workspace_ptr", pos + s_workspace.size());
        }
    }
    // This for cudalib call or __global__ functions;
    def << fu->comment_unit->get_code();
    def << sig << "\n";
    def.block_begin();
    def << body_unit << "\n";
    def.block_end();

#ifdef __USING_HOST_CALL_FORMAT___
    // Turn to Host Call Format in Kernel Definition
    int pos = sig.find(" __global__ ");
    if (pos >= 0)
    {
        pos = sig.find("void ", pos);
        NNFUSION_CHECK(pos >= 0);
        pos += sizeof("void ") - 1;

        auto host_sig = sig.substr(pos);
        pos = host_sig.find("(");
        NNFUSION_CHECK(pos >= 0);

        auto func_name = host_sig.substr(0, pos);
        auto args = host_sig.substr(pos + 1);
        NNFUSION_CHECK(args[args.size() - 1] == ')');
        def << "extern void " << func_name
            << "_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, "
            << args << " {\n";

        args[args.size() - 1] = ',';

        std::vector<std::string> params;
        for (int i = 0, j; j = args.find(',', i), j >= 0; i = j + 1)
        {
            int start = args.find_last_of(' ', j) + 1;
            NNFUSION_CHECK(start >= 1);
            params.push_back(args.substr(start, j - start));
        }

        def << "    " << func_name << "<<<grids, blocks, mem, stream>>>(" << join(params, ", ")
            << ");\n";
        def << "}\n";
    }
#endif

    LanguageUnit dec("dec");
    {
        if (sig.find("extern ") != 0)
            sig = "extern " + sig;
#ifdef __USING_HOST_CALL_FORMAT___
        // Turn to Host Call Format in Kernel Declaration
        int pos = sig.find("__global__");
        if (pos >= 0)
        {
            pos = sig.find("void ", pos);
            NNFUSION_CHECK(pos >= 0);
            int comma = sig.find('(', pos);
            NNFUSION_CHECK(comma >= 0);

            sig =
                "extern " + sig.substr(pos, comma - pos) +
                "_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, " +
                sig.substr(comma + 1);
        }
#endif
        dec << "\n" << sig << ";\n";
    }

    string sname = fu->name_unit->get_code();
    def.change_symbol(sname);
    auto func_def = make_shared<CPUFunctionFile>(dec.get_code(), lu);
    func_def->header_code = func_def->get_extern_declare();
    func_def->source_code = func_def->get_code();
    func_def->extern_decl_unit = std::make_shared<LanguageUnit>(
        func_def->symbol + "_extern_decl_unit", func_def->get_extern_declare());
    return func_def;
}

CPUFunctionFile_p
    CPUFunctionFile::convert_from(std::shared_ptr<nnfusion::kernels::KernelEmitter> kernel)
{
    FunctionUnit_p fu = kernel->get_or_emit_source();
    LanguageUnit_p lu = make_shared<LanguageUnit>();
    LanguageUnit& def = *lu;
    def.require(header::assert);
    def.require(header::stdexcept);
    def.require(header::sstream);
    def.require(header::fstream);
    def.require(header::thread);

    if (kernel->is_parallelism())
        def.require(header::threadpool);

    for (auto& sym : fu->dep_unit->local_symbol)
        def.require(sym.second);

    def << fu->comment_unit->get_code();
    string sig = fu->get_specialized_signature();

    def << sig << "\n";
    def.block_begin();
    def << fu->body_unit->get_code() << "\n";
    def.block_end();

    LanguageUnit dec("dec");
    {
        if (sig.find("extern ") != 0)
            sig = "extern " + sig;
        dec << "\n" << sig << ";\n";
    }

    string sname = fu->name_unit->get_code();
    def.change_symbol(sname);
    auto func_def = make_shared<CPUFunctionFile>(dec.get_code(), lu);
    func_def->header_code = func_def->get_extern_declare();
    func_def->source_code = func_def->get_code();
    func_def->extern_decl_unit = std::make_shared<LanguageUnit>(
        func_def->symbol + "_extern_decl_unit", func_def->get_extern_declare());
    return func_def;
}

void CPUFunctionFile::save_file()
{
    LanguageUnit def_re("require");
    def_re << boilerplate::MIT1->get_code();

    // Write Dependency
    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("header::") != string::npos)
        {
            if (it.second->symbol.find("header::reference_common") != string::npos)
            {
                def_re << R"(
#include "../reference_common.h"
using namespace reference_common;
)";
            }
            else
            {
                def_re << it.second->get_code();
            }
        }
    def_re << "#include<cstring>\n";
    def_re << "using namespace std;\n";
    def_re << "\n";

    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("macro::") != string::npos)
            def_re << it.second->get_code();
    def_re << "\n";
    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("declaration::") != string::npos)
            def_re << it.second->get_code();
    def_re << "\n";
    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("cpu_reference_") != string::npos)
            def_re << it.second->get_code();
    def_re << "\n";

    string fname = this->get_symbol();
    if (fname.length() > 128)
    {
        size_t hashcode = std::hash<std::string>{}(fname);
        fname = "kernels/compressed_src_" + std::to_string(hashcode) + suffix_str;
    }
    else
        fname = "kernels/" + this->get_symbol() + suffix_str;

    ofstream src(fname);
    src << def_re.get_code();
    src << "\n";
    src << this->get_code();
    src.close();
}

HLSLFunctionFile_p
    HLSLFunctionFile::convert_from(std::shared_ptr<nnfusion::kernels::KernelEmitter> kernel)
{
    FunctionUnit_p fu = kernel->get_or_emit_source();
    LanguageUnit_p lu = make_shared<LanguageUnit>();
    LanguageUnit& def = *lu;

    for (auto& sym : fu->dep_unit->local_symbol)
        def.require(sym.second);

    def << fu->comment_unit->get_code();
    string sig = fu->get_specialized_signature();

    if (FLAGS_fhlsl_codegen_type == "csharp")
        def << "static ";
    def << sig << "\n";
    def.block_begin();
    def << fu->body_unit->get_code() << "\n";
    def.block_end();

    LanguageUnit dec("dec");
    {
        if (sig.find("extern ") != 0)
            sig = "extern " + sig;
        dec << "\n" << sig << ";\n";
    }

    string sname = fu->name_unit->get_code();
    def.change_symbol(sname);
    auto func_def = make_shared<HLSLFunctionFile>(dec.get_code(), lu);
    func_def->header_code = func_def->get_extern_declare();
    func_def->source_code = func_def->get_code();
    func_def->extern_decl_unit = std::make_shared<LanguageUnit>(
        func_def->symbol + "_extern_decl_unit", func_def->get_extern_declare());
    return func_def;
}
