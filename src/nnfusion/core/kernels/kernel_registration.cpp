// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "kernel_registration.hpp"
#include "nnfusion/common/type/element_type.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

KernelRegistration::KernelRegistration(const string op_name)
    : m_op_name(op_name)
{
}

KernelRegistration& KernelRegistration::Device(const NNFusion_DeviceType device_type)
{
    m_device_type = device_type;
    return *this;
}

KernelRegistration& KernelRegistration::TypeConstraint(const element::Type data_type)
{
    m_data_type = data_type;
    return *this;
}

KernelRegistration& KernelRegistration::Tag(const string tag)
{
    m_tag = tag;
    return *this;
}

KernelRegistration& KernelRegistration::Priority(size_t priority)
{
    m_priority = priority;
    return *this;
}

KernelRegistration& KernelRegistration::KernelFactory(const Factory factory)
{
    m_factory = factory;
    return *this;
}

const shared_ptr<KernelRegistration> KernelRegistration::Build()
{
    NNFUSION_CHECK(!m_op_name.empty());
    NNFUSION_CHECK_NOT_NULLPTR(m_factory);
    shared_ptr<KernelRegistration> sptr(this);

    return sptr;
}

bool KernelRegistry::RegisterKernel(const string op_name,
                                    shared_ptr<KernelRegistration> registration)
{
    m_kernel_registry.insert(std::make_pair(op_name, registration));
    // NNFUSION_LOG(INFO) << "Registered kernel for Opeartor: " << op_name
    //                    << ", tag: " << registration->m_tag
    //                    << " , dev type: " << nnfusion::get_device_str(registration->m_device_type);

    return true;
}

shared_ptr<const KernelRegistration>
    KernelRegistry::KernelSelect(std::vector<shared_ptr<const KernelRegistration>>& matched_regs)
{
    NNFUSION_CHECK(matched_regs.size() > 0);

    // a naive selector to always return the first matched kernel
    return matched_regs[0];
}

shared_ptr<const KernelRegistration> KernelRegistry::FindKernelRegistration(
    const string op_name, const NNFusion_DeviceType& device_type, const element::Type data_type)
{
    std::vector<shared_ptr<const KernelRegistration>> matched_regs;
    auto regs = m_kernel_registry.equal_range(op_name);
    for (auto iter = regs.first; iter != regs.second; ++iter)
    {
        shared_ptr<const KernelRegistration> reg = iter->second;
        if (device_type == reg->m_device_type && data_type == reg->m_data_type)
        {
            matched_regs.push_back(reg);
        }
    }

    if (matched_regs.size() > 0)
    {
        auto reg = KernelSelect(matched_regs);
        return reg;
    }
    else
    {
        return nullptr;
    }
}

std::vector<shared_ptr<const KernelRegistration>> KernelRegistry::FindKernelRegistrations(
    const string op_name, const NNFusion_DeviceType& device_type, const element::Type data_type)
{
    std::vector<shared_ptr<const KernelRegistration>> matched_regs;
    auto regs = m_kernel_registry.equal_range(op_name);
    for (auto iter = regs.first; iter != regs.second; ++iter)
    {
        shared_ptr<const KernelRegistration> reg = iter->second;
        if (device_type == reg->m_device_type && data_type == reg->m_data_type)
        {
            matched_regs.push_back(reg);
        }
    }

    return matched_regs;
}