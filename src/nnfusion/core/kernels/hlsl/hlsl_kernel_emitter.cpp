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

    if (FLAGS_fhlsl_codegen_type != "default")
    {
        vector<string> names;
        names.insert(names.end(), m_context->input_names.begin(), m_context->input_names.end());
        names.insert(names.end(), m_context->output_names.begin(), m_context->output_names.end());
        names.insert(names.end(), m_context->tensor_names.begin(), m_context->tensor_names.end());

        lu << join(names, ", ");
    }
    else
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