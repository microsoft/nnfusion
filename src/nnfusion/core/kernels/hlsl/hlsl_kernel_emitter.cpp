// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hlsl_kernel_emitter.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "nnfusion/util/errors.hpp"
#include "nnfusion/util/logging.hpp"
using namespace nnfusion;
using namespace nnfusion::kernels;
DECLARE_string(fhlsl_codegen_type);

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

LanguageUnit_p hlsl::AntaresHLSLKernelEmitter::emit_function_call()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
    auto& lu = *_lu;

    if (FLAGS_fhlsl_codegen_type == "cpp")
    {
        std::string kernel_name = get_function_name() + "_kernel_names";
        std::string kernel_names_args = "std::string " + kernel_name + "[] = { ";
        std::string args_name = get_function_name() + "_args";
        std::string args = "void** " + args_name + "[] = { ";
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
            module_args += " };\n";
        }
        kernel_names_args += " };\n";
        args += " };\n";

        lu << kernel_names_args;
        lu << module_args;
        lu << args;
        lu << "dxModuleLaunchAsync(" << get_function_name() << "_module, " << kernel_name << ", "
           << args_name << ", "
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
            lu << "NNfusionTensor ts_" << m_context->output_names[0] << "(device, {"
               << nnfusion::codegen::join_collections(
                      curr->get_output_shape(0),
                      [](int idx, ssize_t it) { return std::to_string(it); })
               << "}, sizeof(" << curr->get_output_element_type(0).c_type_string() << "));\n";

            lu << "  NNfusionOperator op_" << m_context->output_names[0] << "(device, {";
            for (int i = 0; i < curr->get_input_size(); ++i)
            {
                if (i)
                    lu << ", ";
                lu << "ts_" << m_context->input_names[i];
            }
            lu << "}, { ts_" << m_context->output_names[0] << " }, L\"" << get_function_name()
               << ".hlsl\");\n\n";
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
                    int idx = std::atoi(name.substr(5).data());
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