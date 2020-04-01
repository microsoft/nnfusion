// Microsoft (c) 2019, MSRA/NNFUSION Team
/**
 * \brief Profiler::CudaRuntime for creating a Compiler to profile the kernel
 * \author wenxh
 */

#include "cpu_runtime.hpp"
#include <cstdio>
#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/cpu/reference/reference_common.hpp"

using namespace nnfusion::profiler;
using namespace nnfusion::kernels;

bool ReferenceRuntime::codegen(const ProfilingContext::Pointer& ke)
{
    if (ke->source_code != nullptr)
        return true;
    FunctionUnit_p fu = ke->kernel->get_or_emit_source();
    LanguageUnit writer(fu->name_unit->get_code() + ".cpp");
    writer << "// Microsoft (c) 2019, NNFusion\n";

    auto re = fu->dep_unit;
    re->require(header::assert);
    re->require(header::stdexcept);
    re->require(header::sstream);
    re->require(header::fstream);
    re->require(header::thread);
    re->require(declaration::typedef_int);

    // Write Dependency
    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("header::") != string::npos)
        {
            if (it.second->symbol == "header::reference_common")
            {
                writer << "// Unfolded reference_common.h begins\n";
                writer << reference_common_header->get_code();
                writer << "using namespace reference_common;\n";
                writer << "// Unfolded reference_common.h ends\n";
            }
            else
                writer << it.second->get_code();
        }
    writer << "\n";

    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("macro::") != string::npos)
            writer << it.second->get_code();

    writer << "#include <chrono>\n#include<ctime>\n#include<ratio>\n";
    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("declaration::") != string::npos)
            writer << it.second->get_code();
    writer << "\n";

    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("cpu_reference_") != string::npos)
            writer << it.second->get_code();
    writer << "\n";

    //Write Code

    // Write function definition
    writer << fu->comment_unit->get_code();
    writer << fu->get_specialized_signature() << "\n";
    writer.block_begin();
    writer << fu->body_unit->get_code() << "\n";
    writer.block_end();

    auto& arg = ke->kernel->m_context->inputs;
    auto& out = ke->kernel->m_context->outputs;
    auto& temp = ke->kernel->m_context->tensors;

    writer << "extern \"C\" double " << fu->name_unit->get_code() << "_host(";
    for (size_t i = 0; i + 1 < arg.size(); i++)
    {
        writer << arg[i]->get_element_type().c_type_string() << "* " << arg[i]->get_name() << ", ";
    }
    if (!arg.empty())
    {
        writer << arg.back()->get_element_type().c_type_string() << "* " << arg.back()->get_name();
        if (!out.empty())
            writer << ", ";
    }

    for (size_t i = 0; i + 1 < out.size(); i++)
    {
        writer << out[i]->get_element_type().c_type_string() << "* " << out[i]->get_name() << ", ";
    }
    if (!out.empty())
    {
        writer << out.back()->get_element_type().c_type_string() << "* " << out.back()->get_name();
    }
    writer << ")\n";

    auto tensor_declare = [](const shared_ptr<nnfusion::descriptor::Tensor>& t) -> std::string {
        return t->get_element_type().c_type_string() + "* " + t->get_name() + ";\n";
    };

    auto tensor_alloc_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
        stringstream s;
        s << tensor->get_name() << " = new " << tensor->get_element_type().c_type_string() << "["
          << tensor->size(false) << "];\n";
        return s.str();
    };

    auto tensor_free_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
        stringstream s;
        s << "delete[] " << tensor->get_name() << ";\n";
        return s.str();
    };

    writer.block_begin();
    {
        for (size_t i = 0; i < temp.size(); i++)
        {
            writer << tensor_declare(temp[i]);
            writer << tensor_alloc_host(temp[i]);
        }

        writer << "std::chrono::high_resolution_clock::time_point t1,t2;\n";
        writer << "for(int i=0; i < " << ke->warmup_times + ke->runtime_times << "; i++)\n";
        writer.block_begin();
        {
            writer << "if(i == " << ke->warmup_times
                   << ") t1 = std::chrono::high_resolution_clock::now();\n";
            writer << fu->name_unit->get_code() << fu->call_unit->get_code();
        }
        writer.block_end();
        writer << "t2 = std::chrono::high_resolution_clock::now();\n";
        writer << "std::chrono::duration<double> time_span = "
                  "std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);\n";
        writer << "double milliseconds = time_span.count();\n";
        writer << "return milliseconds/" << ke->runtime_times << ";\n";

        for (size_t i = 0; i < temp.size(); i++)
            writer << tensor_free_host(temp[i]);
    }
    writer.block_end();

    writer << "\n";

    writer << "extern \"C\" double " << fu->name_unit->get_code()
           << "_entry(void** args, void** outputs)";

    writer.block_begin();
    {
        writer << "return " << fu->name_unit->get_code() << "_host(";
        for (size_t i = 0; i + 1 < arg.size() + out.size(); i++)
        {
            string type = i < arg.size()
                              ? arg[i]->get_element_type().c_type_string()
                              : (i - arg.size() < out.size()
                                     ? out[i - arg.size()]->get_element_type().c_type_string()
                                     : "");
            writer << "(" << type << "*)" << (i < arg.size() ? "args" : "outputs") << "["
                   << i - (i >= arg.size() ? arg.size() : 0) << "], ";
        }
        if (arg.size() + out.size() > 0)
        {
            int i = arg.size() + out.size() - 1;
            string type = i < arg.size()
                              ? arg[i]->get_element_type().c_type_string()
                              : (i - arg.size() < out.size()
                                     ? out[i - arg.size()]->get_element_type().c_type_string()
                                     : "");
            writer << "(" << type << "*)" << (out.size() == 0 ? "args" : "outputs") << "["
                   << i - (i >= arg.size() ? arg.size() : 0) << "]";
        }
        writer << ");\n";
    }
    writer.block_end();

    ke->source_code = make_shared<LanguageUnit>(move(writer));
    return true;
}

bool ReferenceRuntime::compile(const ProfilingContext::Pointer& ke)
{
    if (ke->entry_point != nullptr)
        return true;
    string filename = string(tmpnam(nullptr));
    string objname = filename + DLIB_SUFFIX;
    string srcname = filename + ".cpp";
    // ofstream source_file(ke->working_dir + "/" + ke->source_code->symbol);
    ofstream source_file(srcname);
    source_file << ke->source_code->get_code();
    source_file.close();

    int ret = system(("gcc\t-fPIC\t-shared\t-std=c++11\t" + srcname + "\t-o\t" + objname).c_str());
    if (ret != 0)
        return false;
    if (!file_exsits(objname))
        return false;
    auto obj = get_library_handle(objname);
    auto entry = get_funcion_pointer(
        ke->kernel->get_or_emit_source()->name_unit->get_code() + "_entry", obj);
    if (entry == nullptr)
        return false;
    ke->entry_point = (double (*)(void**, void**))entry;
    return true;
}

double ReferenceRuntime::invoke(const ProfilingContext::Pointer& ke, void** input, void** output)
{
    // Replacing Existed Kernel with Reference Kenel
    auto& gnode = ke->kernel->m_context->gnode;
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(
            gnode->get_op_type(), GENERIC_CPU, DT_FLOAT);
    shared_ptr<KernelContext> ctx(new KernelContext(gnode));

    bool has_valid_kernel = false;
    for (auto kernel_reg : kernel_regs)
    {
        if (kernel_reg->m_tag != "reference")
            continue;
        auto kernel = kernel_reg->m_factory(ctx);
        if (kernel->get_or_emit_source())
        {
            has_valid_kernel = true;
            LOG(INFO) << "Replacing with reference kenel.";
            // Replacing the kernel;
            ke->kernel = kernel;
        }
    }

    if (!has_valid_kernel)
        return -1.0;
    if (codegen(ke) == false)
        return -1.0;
    if (compile(ke) == false)
        return -1.0;
    if (ke->entry_point == nullptr)
        return -1.0;
    return ke->entry_point(input, output);
}

ReferenceRuntime::Pointer ReferenceRuntime::Runtime()
{
    static ReferenceRuntime::Pointer predefined = nullptr;
    if (predefined == nullptr)
        predefined = make_shared<ReferenceRuntime>();
    return predefined;
}