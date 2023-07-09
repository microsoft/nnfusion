// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cpu_helper.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion::kernels;

LanguageUnit_p cpu::get_eigen_math_kernel(const std::string& name,
                                          const std::string& math_kernel,
                                          size_t data_size,
                                          const std::vector<std::string>& data_types)
{
    NNFUSION_CHECK(std::count(name.begin(), name.end(), '-') == 0);
    std::string mangled_name = "declaration::function_def_inline_" + name;
    // TODO: handle data_types containing underline, like long_long
    // Output type should be ignore
    for (size_t i = 0; i < data_types.size() - 1; i++)
    {
        mangled_name += "-" + data_types[i];
    }
    mangled_name += "-" + std::to_string(data_size);
    shared_ptr<LanguageUnit> cw(new LanguageUnit(mangled_name));
    auto& writer = *cw;
    if (math_kernel.size())
    {
        auto num_inputs = data_types.size() - 1;
        writer << "inline void " << name << "_" << data_size
               << "(concurrency::ThreadPool* thread_pool, ";
        for (size_t i = 0; i < num_inputs; ++i)
        {
            writer << data_types[i] << "* x" << i << ", ";
        }
        writer << data_types[num_inputs] << "* y0";
        writer << ")\n";
        writer << "{\n";
        writer.indent++;
        {
            for (size_t i = 0; i < num_inputs; ++i)
            {
                if (math_kernel.find("pow") != std::string::npos)
                {
                    writer << "Eigen::Map<Eigen::Array<" << data_types[i] << ", 1, " << data_size
                           << ">>"
                           << " in" << i << "(x" << i << ", {" << data_size << "});\n";
                }
                else
                {
                    writer << "Eigen::TensorMap<Eigen::Tensor<" << data_types[i]
                           << ", 1, Eigen::RowMajor>>"
                           << " in" << i << "(x" << i << ", {" << data_size << "});\n";
                }
            }
            if (math_kernel.find("pow") != std::string::npos)
            {
                writer << "Eigen::Map<Eigen::Array<" << data_types[num_inputs] << ", 1, "
                       << data_size << ">>"
                       << " out(y0, {" << data_size << "});\n";
                writer << "out = " << math_kernel << ";\n";
            }
            else
            {
                writer << "Eigen::TensorMap<Eigen::Tensor<" << data_types[num_inputs]
                       << ", 1, Eigen::RowMajor>>"
                       << " out(y0, {" << data_size << "});\n";
                writer << "out.device(*(thread_pool->GetDevice())) = " << math_kernel << ";\n";
            }
        }
        writer.indent--;
        writer << "}\n";
    }
    return cw;
}

LanguageUnit_p cpu::get_simd_math_kernel(const std::string& name,
                                         const std::string& math_kernel,
                                         size_t data_size,
                                         const std::vector<std::string>& data_types)
{
    NNFUSION_CHECK(std::count(name.begin(), name.end(), '-') == 0);
    std::string mangled_name = "declaration::function_def_inline_" + name;
    // TODO: handle data_types containing underline, like long_long
    // Output type should be ignore
    for (size_t i = 0; i < data_types.size() - 1; i++)
    {
        mangled_name += "-" + data_types[i];
    }
    shared_ptr<LanguageUnit> cw(new LanguageUnit(mangled_name));
    auto& writer = *cw;
    if (math_kernel.size())
    {
        auto num_inputs = data_types.size() - 1;
        writer << "inline __m256 " << name << "(";
        for (size_t i = 0; i < num_inputs - 1; ++i)
        {
            writer << "__m256 in" << i << ", ";
        }
        writer << "__m256 in" << num_inputs - 1;
        writer << ")\n";
        writer << "{\n";
        writer.indent++;
        {
            writer << "return " << math_kernel << ";\n";
        }
        writer.indent--;
        writer << "}\n";
    }
    return cw;
}
