// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cpu_kernel_emitter.hpp"
#include <cstring>
#include <sstream>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/util/logging.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

LanguageUnit_p cpu::EigenKernelEmitter::emit_eigen_utils()
{
    LanguageUnit_p _lu(new LanguageUnit("eigen_utils.hpp"));
    auto& lu = *_lu;

    lu << R"(
#pragma once

#include <Eigen/Core>

using DynamicStrides = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
using VectorStrides = Eigen::Stride<Eigen::Dynamic, 1>;

template <typename T>
using DynamicArray =
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using EigenArrayBase = Eigen::Map<DynamicArray<T>, 0, DynamicStrides>;

template <typename T>
using DynamicMatrix =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using EigenMatrixBase = Eigen::Map<DynamicMatrix<T>, 0, DynamicStrides>;

template <typename T>
using DynamicVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using EigenVectorBase = Eigen::Map<DynamicVector<T>, 0, VectorStrides>;


template <typename T>
class EigenArray1d : public EigenArrayBase<T>
{
public:
    EigenArray1d(T* t, size_t s) : EigenArrayBase<T>(t, s, 1, DynamicStrides(1, 1)){}

    template <typename U>
    EigenArray1d& operator=(const U& other)
    {
        EigenArrayBase<T>::operator=(other);
        return *this;
    }
};

template <typename T>
class EigenArray2d : public EigenArrayBase<T>
{
public:
    EigenArray2d(T* t, size_t m, size_t n, size_t s_m, size_t s_n) : 
        EigenArrayBase<T>(t, m, n, DynamicStrides(s_m, s_n)){}

    template <typename U>
    EigenArray2d& operator=(const U& other)
    {
        EigenArrayBase<T>::operator=(other);
        return *this;
    }
};

template <typename T>
class EigenVector : public EigenVectorBase<T>
{
public:
    EigenVector(T* t, size_t s) : EigenVectorBase<T>(t, s, 1, VectorStrides(1, 1)){}

    template <typename U>
    EigenVector& operator=(const U& other)
    {
        EigenVectorBase<T>::operator=(other);
        return *this;
    }
};

template <typename T>
class EigenMatrix : public EigenMatrixBase<T>
{
public:
    EigenMatrix(T* t, size_t m, size_t n, size_t s_m, size_t s_n): 
        EigenMatrixBase<T>(t, m, n, DynamicStrides(s_m, s_n)){}

    template <typename U>
    EigenMatrix& operator=(const U& other)
    {
        EigenMatrixBase<T>::operator=(other);
        return *this;
    }
};
)";
    return _lu;
}

static string format_name(const string& name)
{
    string rc;
    if (!name.empty())
    {
        rc = " " + name;
    }
    return rc;
}

std::string
    cpu::EigenKernelEmitter::emit_eigen_vector(const shared_ptr<nnfusion::descriptor::Tensor> tw,
                                               const string& name)
{
    stringstream ss;

    const element::Type& et = tw->get_element_type();
    ss << "EigenVector<" << et.c_type_string() << ">" << format_name(name) << "(" << tw->get_name()
       << ", " << tw->size(false) << ")";
    return ss.str();
}

std::string
    cpu::EigenKernelEmitter::emit_eigen_matrix(const shared_ptr<nnfusion::descriptor::Tensor> tw,
                                               const string& name)
{
    stringstream ss;

    const element::Type& et = tw->get_element_type();
    ss << "EigenMatrix<" << et.c_type_string() << ">" << format_name(name) << "(" << tw->get_name()
       << ", " << join(tw->get_shape()) << ", " << join(tw->get_tensor_layout()->get_strides())
       << ")";
    return ss.str();
}

LanguageUnit_p cpu::AntaresCpuKernelEmitter::emit_function_body()
{
    auto& ctx = m_context;

    if (antares_code.empty())
        return nullptr;

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // extract kernel code
    const char* s_func_pattern = "// [thread_compute]\n";
    const char* e_func_pattern = "\n}\n";
    const char* s_rank_pattern = "__rank__ = ";
    const char* e_rank_pattern = "\n";
    std::string::size_type s_func_pos = antares_code.find(s_func_pattern);
    std::string::size_type e_func_pos = antares_code.rfind(e_func_pattern);

    if (s_func_pos != std::string::npos || e_func_pos != std::string::npos)
        return nullptr;

    NNFUSION_CHECK(s_func_pos != std::string::npos && e_func_pos != std::string::npos);

    std::string func_body = antares_code.substr(s_func_pos + strlen(s_func_pattern),
                                                e_func_pos - s_func_pos - strlen(s_func_pattern));
    std::string::size_type s_rank_pos = func_body.find(s_rank_pattern);
    std::string::size_type e_rank_pos = func_body.find(e_rank_pattern);
    std::string rank_str = func_body.substr(s_rank_pos + strlen(s_rank_pattern),
                                            e_rank_pos - s_rank_pos - strlen(s_rank_pattern));
    int rank = atoi(rank_str.c_str());
    auto code = op::create_code_from_template(
        R"(
int32_t rank = @rank@;

auto func = [&](int __rank__)
    {
        @func_body@
    };

thread_pool->ParallelFor(rank, func);
)",
        {{"rank", rank}, {"func_body", func_body}});

    lu.block_begin();
    if (func_body.find("min") != std::string::npos)
    {
        lu << "using std::min;\n";
    }
    if (func_body.find("max") != std::string::npos)
    {
        lu << "using std::max;\n";
    }
    lu << code << "\n";
    lu.block_end();
    return _lu;
}

LanguageUnit_p cpu::AntaresCpuKernelEmitter::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    return _lu;
}

bool cpu::AntaresCpuKernelEmitter::is_eliminative()
{
    return (is_memcpy && m_context->inputs[0]->is_same_address(m_context->outputs[0]));
}

LanguageUnit_p cpu::CpuKernelEmitter::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
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
        // defult name is: "persit0", "persist1" ...
        ss << m_context->tensors[i]->get_name();
        params.push_back(ss.str());
    }

    lu << "void (";
    if (this->is_parallelism())
        lu << "concurrency::ThreadPool* thread_pool, ";
    lu << join(params, ", ") << ")";
    return _lu;
}
