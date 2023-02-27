// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/common/languageunit.hpp"

#include <list>

DECLARE_bool(fmem_trace);

namespace nnfusion
{
    class MemoryAllocator
    {
        friend class MemoryAllocatorFactory;

    public:
        enum class block_state
        {
            FREE,
            ALLOCATED
        };

        enum class allocation_scheme
        {
            FIRST_FIT,
            BEST_FIT,
            NO_REUSE
        };

        class node
        {
        public:
            node(size_t size, block_state state);

            bool is_free() const { return m_state == block_state::FREE; }
            size_t m_size;
            block_state m_state;
        };

        // allocate a set of tensors.
        virtual void allocate(std::vector<shared_ptr<descriptor::Tensor>>& tensors);
        // allocate one tensor.
        virtual void allocate(shared_ptr<descriptor::Tensor> tensor);
        // allocate tensor with specified offset.
        virtual void allocate(shared_ptr<descriptor::Tensor> tensor,
                              shared_ptr<descriptor::Tensor> root_tensor,
                              size_t offset = 0);
        void register_tensor(shared_ptr<descriptor::Tensor> tensor);
        virtual void free(shared_ptr<descriptor::Tensor> tensor);

        void dump(std::ofstream&);
        void record(string symbol, shared_ptr<descriptor::Tensor> tensor);
        virtual LanguageUnit_p emit_memory_init();
        virtual LanguageUnit_p emit_memory_alloc();
        virtual LanguageUnit_p emit_memory_free();
        virtual LanguageUnit_p emit_memory_set(int value = 0);
        virtual LanguageUnit_p emit_memory_pool_offset(size_t offset);

        static size_t align(size_t x, size_t alignment);

        std::list<node>::iterator begin() { return m_node_list.begin(); }
        std::list<node>::iterator end() { return m_node_list.end(); }
        std::list<node>::const_iterator begin() const { return m_node_list.cbegin(); }
        std::list<node>::const_iterator end() const { return m_node_list.cend(); }
        const std::list<node>& get_node_list() const { return m_node_list; }
        size_t max_allocated() const { return m_max_allocated; }
        size_t max_alloc_unit() const { return m_max_alloc_unit; }
        size_t cur_allocated() const;
        size_t memory_in_use() const;
        void set_alloc_scheme(allocation_scheme alloc_schem) { m_scheme = alloc_schem; }
        allocation_scheme get_alloc_scheme() const { return m_scheme; }
        void set_alignment(size_t alignment) { m_alignment = alignment; }
        size_t get_alignment() const { return m_alignment; }
        const std::string& get_symbol() const { return m_symbol; }
        const std::string& get_name() const { return m_name; }
        size_t get_device_id() const { return m_device_id; }
        NNFusion_DeviceType get_device_type() const { return m_device_type; }
    protected:
        size_t first_fit(size_t size);
        size_t best_fit(size_t size);
        size_t no_reuse_allocator(size_t size);

        std::list<node> m_node_list;
        size_t m_alignment;
        allocation_scheme m_scheme;
        NNFusion_DeviceType m_device_type;
        size_t m_device_id;
        size_t m_max_allocated;
        std::vector<shared_ptr<descriptor::Tensor>> m_allocated_tensors;
        std::stringstream m_trace;
        bool record_trace = FLAGS_fmem_trace;
        std::string m_symbol;
        std::string m_name;
        size_t m_max_alloc_unit;
        MemoryAllocator(size_t alignment = 1,
                        bool disable_reuse = false,
                        NNFusion_DeviceType device_type = CUDA_GPU,
                        size_t device_id = 0,
                        const std::string& symbol = "");
    };

    class CUDAMemoryAllocator : public MemoryAllocator
    {
        friend class MemoryAllocatorFactory;

    private:
        CUDAMemoryAllocator(size_t alignment = 1,
                            bool disable_reuse = false,
                            NNFusion_DeviceType device_type = CUDA_GPU,
                            size_t device_id = 0,
                            const std::string& symbol = "")
            : MemoryAllocator(alignment, disable_reuse, device_type, device_id, symbol)
        {
        }
    };

    class HostMemoryAllocator : public MemoryAllocator
    {
        friend class MemoryAllocatorFactory;

    public:
        LanguageUnit_p emit_memory_alloc() override;
        LanguageUnit_p emit_memory_free() override;
        LanguageUnit_p emit_memory_set(int value = 0) override;

    private:
        HostMemoryAllocator(size_t alignment = 1,
                            bool disable_reuse = false,
                            NNFusion_DeviceType device_type = GENERIC_CPU,
                            size_t device_id = 0,
                            const std::string& symbol = "")
            : MemoryAllocator(alignment, disable_reuse, device_type, device_id, symbol)
        {
        }
    };

    class RocmMemoryAllocator : public MemoryAllocator
    {
        friend class MemoryAllocatorFactory;

    private:
        RocmMemoryAllocator(size_t alignment = 1,
                            bool disable_reuse = false,
                            NNFusion_DeviceType device_type = ROCM_GPU,
                            size_t device_id = 0,
                            const std::string& symbol = "")
            : MemoryAllocator(alignment, disable_reuse, device_type, device_id, symbol)
        {
        }
    };

    class RDMAMemoryAllocator : public MemoryAllocator
    {
        friend class MemoryAllocatorFactory;

    private:
        RDMAMemoryAllocator(size_t alignment = 1,
                            bool disable_reuse = false,
                            NNFusion_DeviceType device_type = CUDA_GPU,
                            size_t device_id = 0,
                            const std::string& symbol = "RDMA")
            : MemoryAllocator(alignment, disable_reuse, device_type, device_id, symbol)
        {
        }
    };

    class HLSLMemoryAllocator : public MemoryAllocator
    {
        friend class MemoryAllocatorFactory;

    public:
        LanguageUnit_p emit_memory_init() override;
        LanguageUnit_p emit_memory_alloc() override;
        LanguageUnit_p emit_memory_free() override;
        LanguageUnit_p emit_memory_set(int value = 0) override;
        LanguageUnit_p emit_memory_pool_offset(size_t offset) override;

    private:
        HLSLMemoryAllocator(size_t alignment = 1,
                            bool disable_reuse = false,
                            NNFusion_DeviceType device_type = HLSL,
                            size_t device_id = 0,
                            const std::string& symbol = "")
            : MemoryAllocator(alignment, disable_reuse, device_type, device_id, symbol)
        {
        }
    };

    class MemoryAllocatorFactory
    {
    public:
        MemoryAllocatorFactory(size_t alignment = 1, bool disable_reuse = false);
        MemoryAllocator* get_allocator(shared_ptr<descriptor::Tensor> tensor);
        size_t get_alignment() const { return m_alignment; }
        const std::unordered_map<std::string, MemoryAllocator*>& get_allocator_list()
        {
            return m_allocator_list;
        }

    private:
        size_t m_alignment;
        bool m_disable_reuse;
        // map from names to allocators
        std::unordered_map<std::string, MemoryAllocator*> m_allocator_list;
    };
}
