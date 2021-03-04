// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <stdio.h>

#include "hlsl_kernel_emitter.hpp"
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/common_langunit.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_string(fhlsl_codegen_type);
namespace nnfusion
{
    namespace kernels
    {
        namespace hlsl
        {
            class Constant : public HLSLKernelEmitter
            {
            public:
                Constant(shared_ptr<KernelContext> ctx)
                    : HLSLKernelEmitter(ctx)
                {
                    op = static_pointer_cast<nnfusion::op::Constant>(ctx->gnode->get_op_ptr());
                    NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not Constant.";

                    folder = "./Constant/";
                }

                LanguageUnit_p emit_function_body() override
                {
                    nnfusion::codegen::create_folder(folder);
                    const_name = m_context->outputs[0]->get_name() + ".bin";

                    const void* dptr = op->get_data_ptr();
                    size_t size = op->get_data_size();

                    FILE* fp = fopen((folder + const_name).c_str(), "wb");
                    NNFUSION_CHECK(fp != nullptr);
                    NNFUSION_CHECK(size == fwrite(dptr, 1, size, fp));
                    fclose(fp);

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& writer = *_lu;
                    if (FLAGS_fhlsl_codegen_type == "csharp")
                    {
                        writer << "var byteArray = File.ReadAllBytes(path);\n";
                        writer << "dxMemcpyHtoDAsync(output0, "
                                  "Marshal.UnsafeAddrOfPinnedArrayElement(byteArray, 0), bytes, "
                                  "IntPtr.Zero);\n";
                    }
                    else
                    {
                        writer << "std::ifstream bin_file(path, std::ios::in | std::ios::binary);\n"
                               << "if(bin_file.fail())\n"
                               << "{\n"
                               << "\tstd::cout << \"Load Constant failed.\\n\" << std::endl;\n"
                               << "\texit(1);\n"
                               << "}\n"
                               << "char* tmp_mem = new char[bytes];\n"
                               << "bin_file.read(tmp_mem, bytes);\n"
                               << "dxMemcpyHtoDAsync(output0, tmp_mem, bytes, 0);\n"
                               << "bin_file.close();\n";
                    }

                    return _lu;
                }

                LanguageUnit_p emit_function_call() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
                    auto& lu = *_lu;

                    auto curr = m_context->gnode;
                    if (FLAGS_fhlsl_codegen_type != "default")
                    {
                        vector<string> names;
                        names.insert(names.end(),
                                     m_context->input_names.begin(),
                                     m_context->input_names.end());
                        names.insert(names.end(),
                                     m_context->output_names.begin(),
                                     m_context->output_names.end());
                        names.insert(names.end(),
                                     m_context->tensor_names.begin(),
                                     m_context->tensor_names.end());

                        lu << "(\"" << folder + const_name << "\", " << op->get_data_size() << ", "
                           << join(names, ", ") << ");\n";
                    }
                    else
                    {
                        lu << "NNfusionTensor ts_" << m_context->output_names[0] << "(device, {"
                           << nnfusion::codegen::join_collections(
                                  curr->get_output_shape(0),
                                  [](int idx, ssize_t it) { return std::to_string(it); })
                           << "}, sizeof(" << curr->get_output_element_type(0).c_type_string()
                           << "));\n";

                        lu << "  NNfusionMemcpy op_" << m_context->output_names[0] << "(device, ts_"
                           << m_context->output_names[0] << ", load_data<"
                           << curr->get_output_element_type(0).c_type_string() << ">(\""
                           << m_context->output_names[0] << ".bin\", ts_"
                           << m_context->output_names[0] << ".NumElements()), true);\n\n";
                    }
                    return _lu;
                }

                LanguageUnit_p emit_function_signature() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
                    auto& lu = *_lu;

                    vector<string> params;
                    for (size_t i = 0; i < m_context->inputs.size(); i++)
                    {
                        stringstream ss;
                        // ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
                        if (FLAGS_fhlsl_codegen_type == "cpp")
                        {
                            ss << "void* input" << i;
                        }
                        else if (FLAGS_fhlsl_codegen_type == "csharp")
                        {
                            ss << "IntPtr input" << i;
                        }
                        params.push_back(ss.str());
                    }

                    for (size_t i = 0; i < m_context->outputs.size(); i++)
                    {
                        stringstream ss;
                        // ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
                        if (FLAGS_fhlsl_codegen_type == "cpp")
                        {
                            ss << "void* output" << i;
                        }
                        else if (FLAGS_fhlsl_codegen_type == "csharp")
                        {
                            ss << "IntPtr output" << i;
                        }

                        params.push_back(ss.str());
                    }

                    lu << "void "
                       << "(std::string path, int bytes, " << join(params, ", ") << ")";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override

                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::fstream);
                    return _lu;
                }

            private:
                shared_ptr<nnfusion::op::Constant> op;
                string folder;
                string const_name;
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("Constant",
                        Device(HLSL).TypeConstraint(element::f32).Tag("hlsl_kernel"),
                        hlsl::Constant)