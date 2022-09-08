// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hlsl_kernel_emitter.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "nnfusion/util/errors.hpp"
#include "nnfusion/util/logging.hpp"
using namespace nnfusion;
using namespace nnfusion::kernels;
DECLARE_string(fhlsl_codegen_type);
DECLARE_bool(fsymbolic);

LanguageUnit_p hlsl::HLSLKernelEmitter::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    return _lu;
}

LanguageUnit_p hlsl::AntaresHLSLKernelEmitter::emit_function_body()
{
    if (antares_code.empty())
        return nullptr;

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // extract kernel code
    lu << antares_code << "\n";
    return _lu;
}

void hlsl::AntaresHLSLKernelEmitter::find_launch_config(
    const std::string& str, std::map<std::string, std::string>& symbol_expr, std::string& blockNum)
{
    // find dynamic blocks for symbolic inputs
    if (symbol_expr.size() > 0)
    {
        std::vector<std::string> sym_args;
        for (auto& p : symbol_expr)
        {
            sym_args.push_back(p.second);
        }

        int pos = str.find("// [thread_extent] $$ = ");
        int block_base =
            (pos >= 0) ? std::atoi(str.data() + pos + sizeof("// [thread_extent] $$ = ") - 1) : 1;
        if (block_base == -1)
            block_base = 1;

        blockNum = std::to_string(block_base);
        int arg_idx = 0;
        int check_value = block_base;
        while (true)
        {
            std::string target = "// [thread_extent] $" + std::to_string(arg_idx) + " = ";
            pos = str.find(target);
            if (pos == std::string::npos)
                break;
            int value = std::atoi(str.data() + pos + target.size());
            std::string value_str(str.data() + pos + target.size());
            if (value > 0)
            {
                NNFUSION_CHECK(arg_idx < sym_args.size());
                blockNum = blockNum + " * ceil(float(" + sym_args[arg_idx] + ") / " +
                           std::to_string(value) + ")";
            }
            arg_idx++;
        }
    }
}

LanguageUnit_p hlsl::AntaresHLSLKernelEmitter::emit_function_call()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
    auto& lu = *_lu;

    if (FLAGS_fhlsl_codegen_type == "cpp")
    {
        std::string kernel_name = get_function_name() + "_kernel_names";
        std::string kernel_names_args = "std::string " + kernel_name + "[] = { ";
        std::string args_name = get_function_name() + "_args";
        std::string block_name = get_function_name() + "_num_blocks";
        std::string args = "void** " + args_name + "[] = { ";
        std::string blocks = "int " + block_name + "[] = { ";
        std::string module_args;
        for (size_t i = 0; i < kernel_info.size(); i++)
        {
            auto ki = kernel_info[i];
            kernel_names_args += "\"" + ki->kernel_name + "\"";
            std::string module_args_name = get_function_name() + "_module_args_" + to_string(i);
            args += module_args_name;

            if (i != kernel_info.size() - 1)
            {
                kernel_names_args += ", ";
                args += ", ";
            }

            module_args += "void* " + module_args_name + "[] = { ";
            for (size_t j = 0; j < ki->input_names.size(); j++)
            {
                auto in_name = ki->input_names[j];
                module_args += tensor_name_map[in_name] + ", ";
            }
            for (size_t j = 0; j < ki->output_names.size(); j++)
            {
                auto out_name = ki->output_names[j];
                module_args += tensor_name_map[out_name];
                if (j != ki->output_names.size() - 1)
                {
                    module_args += ", ";
                }
            }
            std::string num_block = std::to_string(-1);
            if (FLAGS_fsymbolic)
            {
                // parse symbolic args
                std::map<std::string, std::string> symbol_expr;
                NNFUSION_CHECK(kernel_info.size() == 1)
                    << "Symbolic kenrel currently only support single kernel!";
                for (size_t i = 0; i < m_context->inputs.size(); i++)
                {
                    auto shape = m_context->inputs[i]->get_shape();
                    if (shape.is_dynamic())
                    {
                        for (auto dim : *(shape.get_sym_shape()))
                        {
                            if (dim.is_dynamic())
                            {
                                symbol_expr[dim.expr_to_symbol(dim.sym())] = dim.sym();
                            }
                        }
                    }
                }
                for (size_t i = 0; i < m_context->outputs.size(); i++)
                {
                    auto shape = m_context->outputs[i]->get_shape();
                    if (shape.is_dynamic())
                    {
                        for (auto dim : *(shape.get_sym_shape()))
                        {
                            if (dim.is_dynamic())
                            {
                                symbol_expr[dim.expr_to_symbol(dim.sym())] = dim.sym();
                            }
                        }
                    }
                }

                // the key is sortted by std::map
                for (auto& p : symbol_expr)
                {
                    module_args = module_args + ", (void*)(long)(" + p.second + ")";
                }
                // TODO: for multiple kernel, use corrsponding code instead of full antares code
                find_launch_config(antares_code, symbol_expr, num_block);
            }
            module_args += " };\n";
            blocks += num_block;
            if (i != kernel_info.size() - 1)
            {
                blocks += ", ";
            }
        }
        kernel_names_args += " };\n";
        args += " };\n";
        blocks += " };\n";

        lu << kernel_names_args;
        lu << module_args;
        lu << args;
        lu << blocks;
        lu << "dxModuleLaunchAsync(" << get_function_name() << "_module, " << kernel_name << ", "
           << args_name << ", " << block_name << ", "
           << "std::size(" << args_name << "));\n";
    }
    else if (FLAGS_fhlsl_codegen_type == "csharp")
    {
        vector<string> names;
        names.insert(names.end(), m_context->input_names.begin(), m_context->input_names.end());
        names.insert(names.end(), m_context->output_names.begin(), m_context->output_names.end());
        names.insert(names.end(), m_context->tensor_names.begin(), m_context->tensor_names.end());

        lu << join(names, ", ");
    }
    else // default
    {
        if (int(options.find("|memcpy|")) >= 0)
        {
            NNFUSION_CHECK(m_context->inputs.size() == 1);
            lu << "NNfusionTensor &ts_" << m_context->output_names[0] << " = ts_"
               << m_context->input_names[0] << ";\n\n";
            return _lu;
        }

        lu << "// " << ir << "\n";
        auto curr = m_context->gnode;
        NNFUSION_CHECK_NOT_NULLPTR(curr);

        if (int(options.find("|inplace_wg|")) < 0)
        {
            for (int i = 0; i < curr->get_output_size(); ++i)
            {
                lu << "NNfusionTensor ts_" << m_context->output_names[i] << "(device, {"
                   << nnfusion::codegen::join_collections(
                          curr->get_output_shape(i),
                          [](int idx, ssize_t it) { return std::to_string(it); })
                   << "}, sizeof(" << curr->get_output_element_type(i).c_type_string() << "));\n";
            }

            lu << "  NNfusionOperator op_" << m_context->output_names[0] << "(device, {";
            for (int i = 0; i < curr->get_input_size(); ++i)
            {
                if (i)
                    lu << ", ";
                lu << "ts_" << m_context->input_names[i];
            }
            lu << "}, {";
            for (int i = 0; i < curr->get_output_size(); ++i)
            {
                if (i)
                    lu << ", ";
                lu << "ts_" << m_context->output_names[i];
            }
            lu << " }, L\"" << get_function_name() << ".hlsl\");\n\n";
        }
        else
        {
            lu << "  NNfusionOperator op_" << m_context->output_names[0] << "(device, {";
            for (int i = 0; i < curr->get_input_size(); ++i)
            {
                if (i)
                    lu << ", ";
                lu << "ts_" << m_context->input_names[i];
            }
            lu << "}, { ts_" << m_context->input_names[0] << " }, L\"" << get_function_name()
               << ".hlsl\");\n";
            lu << "auto& ts_" << m_context->output_names[0] << " = ts_" << m_context->input_names[0]
               << ";\n\n";
        }
    }
    return _lu;
}

bool hlsl::AntaresHLSLKernelEmitter::is_eliminative()
{
    return (is_memcpy && m_context->inputs[0]->is_same_address(m_context->outputs[0]));
}

void hlsl::AntaresHLSLKernelEmitter::process_antares_kernel_info()
{
    for (auto ki : kernel_info)
    {
        for (size_t i = 0; i < ki->input_names.size(); i++)
        {
            std::string name = ki->input_names[i];
            if (tensor_name_map.find(name) == tensor_name_map.end())
            {
                if (name.find("input") != std::string::npos)
                {
                    int idx = std::atoi(name.substr(5).data());
                    tensor_name_map[name] = m_context->input_names[idx];
                }
                else if (name.find("mediate") != std::string::npos)
                {
                    std::string dtype_str = ki->input_dtypes[i];
                    element::Type dtype;
                    NNFUSION_CHECK(
                        element::Type::dtype_string_to_nnfusion_element_type(dtype_str, dtype));

                    std::string shape_str = ki->input_shapes[i].substr(1);
                    std::vector<size_t> shape;
                    shape.push_back(std::atoi(shape_str.data()));
                    int pos = shape_str.find(", ");
                    while (pos > 0)
                    {
                        shape_str = shape_str.substr(pos + 2);
                        shape.push_back(std::atoi(shape_str.data()));
                        pos = shape_str.find(", ");
                    }

                    auto tmp_tensor = allocate_tensor(Shape(shape), dtype);
                    tensor_name_map[name] = tmp_tensor->get_name();
                }
            }
        }

        for (size_t i = 0; i < ki->output_names.size(); i++)
        {
            std::string name = ki->output_names[i];
            if (tensor_name_map.find(name) == tensor_name_map.end())
            {
                if (name.find("output") != std::string::npos)
                {
                    int idx = std::atoi(name.substr(6).data());
                    tensor_name_map[name] = m_context->output_names[idx];
                }
                else if (name.find("mediate") != std::string::npos)
                {
                    std::string dtype_str = ki->output_dtypes[i];
                    element::Type dtype;
                    NNFUSION_CHECK(
                        element::Type::dtype_string_to_nnfusion_element_type(dtype_str, dtype));

                    std::string shape_str = ki->output_shapes[i].substr(1);
                    std::vector<size_t> shape;
                    shape.push_back(std::atoi(shape_str.data()));
                    int pos = shape_str.find(", ");
                    while (pos > 0)
                    {
                        shape_str = shape_str.substr(pos + 2);
                        shape.push_back(std::atoi(shape_str.data()));
                        pos = shape_str.find(", ");
                    }

                    auto tmp_tensor = allocate_tensor(Shape(shape), dtype);
                    tensor_name_map[name] = tmp_tensor->get_name();
                }
            }
        }
    }
}