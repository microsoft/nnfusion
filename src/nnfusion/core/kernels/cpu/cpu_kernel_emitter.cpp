// Microsoft (c) 2019, NNFusion Team
#include "cpu_kernel_emitter.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

LanguageUnit_p cpu::EigenKernelEmitter::emit_eigen_utils()
{
    LanguageUnit_p _lu(new LanguageUnit("eigen_utils.hpp"));
    auto& lu = *_lu;

    lu << R"(
// Microsoft (c) 2019, NNFusion Team
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
