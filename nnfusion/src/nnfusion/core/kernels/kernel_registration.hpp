// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "kernel_emitter.hpp"
#include "nnfusion/common/type/element_type.hpp"

namespace nnfusion
{
    namespace kernels
    {
        class KernelRegistration
        {
        public:
            typedef shared_ptr<KernelEmitter> (*Factory)(shared_ptr<KernelContext>);

            // Starts with just the name field set. Required
            explicit KernelRegistration(const string op_name);
            ~KernelRegistration(){};

            // Required: specify the device type (e.g., CUDA_GPU) this kernel supports.
            // Return *this
            KernelRegistration& Device(const NNFusion_DeviceType device_type);

            // Specify the data (inputs/outputs) types this kernel supports
            // Return *this
            KernelRegistration& TypeConstraint(const element::Type data_type);

            // Add an arbitrary user-defined tag on the kernel to allow the operator
            // to choose this kernel
            // Return *this
            KernelRegistration& Tag(const string tag);

            // Specify the priority level of this kernel
            // Return *this
            KernelRegistration& Priority(size_t priority);

            // Required: specify the kernel factory that creates this kernel emitter
            // Return *this
            KernelRegistration& KernelFactory(const Factory factory);

            // The final step to create an kernel emitter registration
            const shared_ptr<KernelRegistration> Build();

            const string op_name() { return m_op_name; }
            void debug_string() const
            {
                NNFUSION_LOG(INFO) << "m_op_name: " << m_op_name << "\n"
                                   << "m_device_type: " << m_device_type << "\n"
                                   << "m_data_type: " << m_data_type << "\n"
                                   << "m_tag: " << m_tag << "\n"
                                   << "m_priority: " << m_priority << "\n"
                                   << "m_factory: " << m_factory;
            }
            void set_priority(size_t priority) { m_priority = priority; }
        public:
            friend class KernelRegistry;
            string m_op_name;
            NNFusion_DeviceType m_device_type;
            element::Type m_data_type;
            string m_tag;
            Factory m_factory;
            size_t m_priority = 0;
        };

        static KernelRegistration& Name(const string op_name)
        {
            // TODO(jxue): managed with a shared ptr
            KernelRegistration* registration = new KernelRegistration(op_name);
            return *registration;
        }

        class KernelRegistry
        {
        public:
            KernelRegistry(){};
            ~KernelRegistry(){};
            bool RegisterKernel(const string op_name, shared_ptr<KernelRegistration> registration);
            shared_ptr<const KernelRegistration>
                FindKernelRegistration(const string op_name,
                                       const NNFusion_DeviceType& device_type,
                                       const element::Type data_type);
            std::vector<shared_ptr<const KernelRegistration>>
                FindKernelRegistrations(const string op_name,
                                        const NNFusion_DeviceType& device_type,
                                        const element::Type data_type);
            shared_ptr<const KernelRegistration>
                KernelSelect(std::vector<shared_ptr<const KernelRegistration>>& matched_regs);

            size_t RegisteredKernelSize() const { return m_kernel_registry.size(); }
            static KernelRegistry* Global()
            {
                static KernelRegistry* global_kernel_registry = new KernelRegistry();
                return global_kernel_registry;
            }

        private:
            std::unordered_multimap<string, shared_ptr<KernelRegistration>> m_kernel_registry;
        };

        class KernelRegistrar
        {
        public:
            KernelRegistrar(const string op_name, shared_ptr<KernelRegistration> registration)
            {
                KernelRegistry::Global()->RegisterKernel(op_name, registration);
            }
        };

#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)

#define REGISTER_KERNEL_EMITTER(op_name, attrs, constructor)                                       \
    static KernelRegistrar CONCAT(kernel_registrar, __COUNTER__)(                                  \
        op_name,                                                                                   \
        Name(op_name)                                                                              \
            .attrs                                                                                 \
            .KernelFactory([](shared_ptr<KernelContext> context) -> shared_ptr<KernelEmitter> {    \
                return make_shared<constructor>(context);                                          \
            })                                                                                     \
            .Build());

    } // namespace kernels
} // namespace nnfusion