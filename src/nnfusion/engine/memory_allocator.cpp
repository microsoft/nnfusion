// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "memory_allocator.hpp"
DECLARE_string(fhlsl_codegen_type);
DECLARE_bool(fcustomized_mem_imp);
DECLARE_bool(ffunction_codegen);
DECLARE_bool(fcodegen_debug_half);
DEFINE_bool(fcuda_set_device, true, "Emit cudaSetDevice call");

nnfusion::MemoryAllocator::node::node(size_t size, block_state state)
    : m_size{size}
    , m_state{state}
{
}

nnfusion::MemoryAllocator::MemoryAllocator(size_t alignment,
                                           bool disable_memory_reuse,
                                           NNFusion_DeviceType device_type,
                                           size_t device_id,
                                           const std::string& symbol)
    : m_alignment{alignment}
    , m_scheme{disable_memory_reuse ? allocation_scheme::NO_REUSE : allocation_scheme::FIRST_FIT}
    , m_device_type(device_type)
    , m_device_id(device_id)
    , m_max_allocated{0}
    , m_max_alloc_unit{0}
    , m_symbol(symbol)
    , m_name(symbol + "_" + get_device_str(device_type) + std::to_string(device_id) + "_allocator")
{
    NNFUSION_CHECK_WITH_EXCEPTION(m_alignment > 0, errors::InvalidArgument)
        << "Memory alignment must be > 0";
    m_node_list.emplace_back(numeric_limits<size_t>::max(), block_state::FREE);
    if (record_trace)
    {
        m_trace << this->get_name() << ": \n";
        m_trace << "memory allocation trace: \n";
    }
}

void nnfusion::MemoryAllocator::allocate(std::vector<shared_ptr<descriptor::Tensor>>& tensors)
{
    size_t rc;
    size_t total_size = 0;

    for (auto tensor : tensors)
    {
        total_size += tensor->size();
    }

    if (total_size > m_max_alloc_unit)
        m_max_alloc_unit = total_size;
    switch (m_scheme)
    {
    case allocation_scheme::FIRST_FIT: rc = first_fit(total_size); break;
    case allocation_scheme::BEST_FIT: rc = best_fit(total_size); break;
    case allocation_scheme::NO_REUSE: rc = no_reuse_allocator(total_size); break;
    }
    for (auto tensor : tensors)
    {
        tensor->set_pool_offset(rc);
        // add tensor allocated by this allocator
        m_allocated_tensors.push_back(tensor);
        rc += tensor->size();
        if (record_trace)
        {
            this->record("[allocate]", tensor);
        }
    }
}

void nnfusion::MemoryAllocator::allocate(shared_ptr<descriptor::Tensor> tensor)
{
    size_t rc;
    size_t size = tensor->size();
    if (size > m_max_alloc_unit)
        m_max_alloc_unit = size;
    switch (m_scheme)
    {
    case allocation_scheme::FIRST_FIT: rc = first_fit(size); break;
    case allocation_scheme::BEST_FIT: rc = best_fit(size); break;
    case allocation_scheme::NO_REUSE: rc = no_reuse_allocator(size); break;
    }
    tensor->set_pool_offset(rc);
    tensor->set_pool(this->get_name());
    // add tensor allocated by this allocator
    m_allocated_tensors.push_back(tensor);
    if (record_trace)
    {
        this->record("[allocate]", tensor);
    }
}

void nnfusion::MemoryAllocator::register_tensor(shared_ptr<descriptor::Tensor> tensor)
{
    m_allocated_tensors.push_back(tensor);
    if (record_trace)
    {
        this->record("[register]", tensor);
    }
}

void nnfusion::MemoryAllocator::allocate(shared_ptr<descriptor::Tensor> tensor,
                                         shared_ptr<descriptor::Tensor> root_tensor,
                                         size_t offset)
{
    tensor->set_pool_offset(offset);
    tensor->set_pool(this->get_name());
    auto root = root_tensor;
    NNFUSION_CHECK(!(root->get_root_tensor()));
    tensor->set_root_tensor(root);
    size_t ref_count = root->ref();
    NNFUSION_CHECK(ref_count > 1);
    m_allocated_tensors.push_back(tensor);

    if (record_trace)
    {
        this->record("[allocate]", tensor);
    }
}

size_t nnfusion::MemoryAllocator::no_reuse_allocator(size_t size)
{
    size_t offset = m_max_allocated;
    m_max_allocated += align(size, m_alignment);
    return offset;
}

size_t nnfusion::MemoryAllocator::best_fit(size_t size)
{
    size = align(size, m_alignment);
    size_t offset = 0;
    size_t min_delta = numeric_limits<size_t>::max();
    auto best_fit = m_node_list.end();
    size_t best_offset = offset;
    for (auto it = m_node_list.begin(); it != m_node_list.end(); ++it)
    {
        if (it->m_state == block_state::FREE && it->m_size >= size)
        {
            size_t delta = it->m_size - size;
            if (delta < min_delta)
            {
                min_delta = delta;
                best_fit = it;
                best_offset = offset;
            }
        }
        offset += it->m_size;
    }

    if (best_fit == m_node_list.end())
    {
        throw bad_alloc();
    }

    if (min_delta == 0)
    {
        // exact fit
        best_fit->m_state = block_state::ALLOCATED;
    }
    else
    {
        m_node_list.insert(best_fit, node{size, block_state::ALLOCATED});
        best_fit->m_size -= size;
    }
    m_max_allocated = max(m_max_allocated, best_offset + size);

    return best_offset;
}

size_t nnfusion::MemoryAllocator::first_fit(size_t size)
{
    size = align(size, m_alignment);
    size_t offset = 0;
    bool found = false;
    for (auto it = m_node_list.begin(); it != m_node_list.end(); ++it)
    {
        if (it->m_state == block_state::FREE && it->m_size >= size)
        {
            if (it->m_size > size)
            {
                m_node_list.insert(it, node{size, block_state::ALLOCATED});
                it->m_size -= size;
            }
            else
            {
                // exact fit
                it->m_state = block_state::ALLOCATED;
            }

            found = true;
            break;
        }
        offset += it->m_size;
    }
    if (!found)
    {
        throw bad_alloc();
    }
    m_max_allocated = max(m_max_allocated, offset + size);

    return offset;
}

void nnfusion::MemoryAllocator::free(shared_ptr<descriptor::Tensor> tensor)
{
    // for ref_tensor, just free its root tensor
    if (tensor->get_root_tensor())
    {
        free(tensor->get_root_tensor());
        if (record_trace)
        {
            this->record("[free]", tensor);
        }
        return;
    }

    if (tensor->deref() > 0)
        return;

    size_t offset = tensor->get_pool_offset();
    size_t search_offset = 0;
    bool found = false;
    for (auto it = m_node_list.begin(); it != m_node_list.end(); ++it)
    {
        if (offset == search_offset)
        {
            list<node>::iterator it_next = next(it);
            if (it == m_node_list.begin())
            {
                // free the first node in the list
                it->m_state = block_state::FREE;
            }
            else
            {
                // node has predecessor
                list<node>::iterator it_prev = prev(it);
                if (it_prev->m_state == block_state::FREE)
                {
                    it->m_size += it_prev->m_size;
                    m_node_list.erase(it_prev);
                }
            }
            if (it_next != m_node_list.end() && it_next->m_state == block_state::FREE)
            {
                // join this node with next
                it->m_size += it_next->m_size;
                m_node_list.erase(it_next);
            }
            it->m_state = block_state::FREE;
            found = true;
            break;
        }
        search_offset += it->m_size;
    }
    if (record_trace)
    {
        this->record("[free]", tensor);
    }
    NNFUSION_CHECK(found) << "bad free";
}

void nnfusion::MemoryAllocator::dump(ofstream& out)
{
    out << m_trace.str();
    out << "max allocated memory:\n" << m_max_allocated << "\n";
    out << "current allocated memory:\n" << this->cur_allocated() << "\n";
    out << "current memory in use: \n" << this->memory_in_use() << "\n";
    out << "memory block state: \n";
    for (const node& n : m_node_list)
    {
        out << "size=" << n.m_size << ", ";
        out << (n.m_state == block_state::FREE ? "FREE" : "ALLOCATED");
        out << "\n";
    }
}

void nnfusion::MemoryAllocator::record(string symbol, shared_ptr<descriptor::Tensor> tensor)
{
    m_trace << symbol << " name: " << tensor->get_name()
            << "  offset: " << tensor->get_pool_offset() << "  size: " << tensor->size() << "\n";
}

LanguageUnit_p nnfusion::MemoryAllocator::emit_memory_init()
{
    LanguageUnit_p _lu(new LanguageUnit("declaration::" + this->get_name() + "_init"));
    auto& lu = *_lu;
    if (m_max_allocated > 0)
    {
        lu << "char* " << this->get_name() << "_memory_pool;\n";

        for (auto tensor : m_allocated_tensors)
        {
            lu << element::get_backend_cstring(tensor->get_element_type()) << "* "
               << tensor->get_name() << ";\n";
        }
    }
    return _lu;
}

LanguageUnit_p nnfusion::MemoryAllocator::emit_memory_alloc()
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_alloc"));
    if (FLAGS_fcustomized_mem_imp)
        return _lu;

    auto& lu = *_lu;
    if (m_max_allocated > 0)
    {
        if (FLAGS_fcuda_set_device)
        {
            lu << "CUDA_SAFE_CALL(cudaSetDevice(" << m_device_id << "));\n";
        }
        if (!FLAGS_ffunction_codegen)
        {
            lu << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << this->get_name() << "_memory_pool,"
               << m_max_allocated << "));\n";
            lu << "CUDA_SAFE_CALL(cudaMemset((void*)" << this->get_name() << "_memory_pool, 0, "
               << m_max_allocated << "));\n";
        }

        for (auto tensor : m_allocated_tensors)
        {
            if (tensor->get_shared_tensor())
            {
                // this tensor can be shared with the grpah_0's tensor
                lu << tensor->get_name() << " = ("
                   << element::get_backend_cstring(tensor->get_element_type())
                   << "*)(graph_0::" << tensor->get_shared_tensor()->get_name() << ");\n";
                continue;
            }
            if (tensor->get_pool_offset() == SIZE_MAX)
            {
                NNFUSION_LOG(NNFUSION_WARNING)
                    << tensor->get_name()
                    << " may refer an external tensor, nnfusion omits its memory allocation.";
                lu << "// ";
            }
            NNFUSION_CHECK(tensor->get_pool() == this->get_name());
            lu << tensor->get_name() << " = ("
               << element::get_backend_cstring(tensor->get_element_type()) << "*)("
               << this->get_name() << "_memory_pool+" << tensor->get_pool_offset() << ");\n";
        }
    }
    return _lu;
}

LanguageUnit_p nnfusion::MemoryAllocator::emit_memory_pool_offset(size_t offset)
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_offset"));
    auto& lu = *_lu;
    lu << this->get_name() << "_memory_pool = (char*)(workspace + " << offset << ");\n";
    return _lu;
}

LanguageUnit_p nnfusion::MemoryAllocator::emit_memory_free()
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_free"));
    if (FLAGS_fcustomized_mem_imp)
        return _lu;

    auto& lu = *_lu;
    lu << "CUDA_SAFE_CALL(cudaSetDevice(" << m_device_id << "));\n";
    if (!FLAGS_ffunction_codegen)
        lu << "CUDA_SAFE_CALL(cudaFree(" << this->get_name() + "_memory_pool));\n";
    return _lu;
}

LanguageUnit_p nnfusion::MemoryAllocator::emit_memory_set(int value)
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_memset"));
    auto& lu = *_lu;
    lu << "CUDA_SAFE_CALL(cudaSetDevice(" << m_device_id << "));\n";
    lu << "CUDA_SAFE_CALL(cudaMemset((void*)" << this->get_name() + "_memory_pool, " << value
       << ", " << m_max_allocated << "));\n";
    return _lu;
}

size_t nnfusion::MemoryAllocator::align(size_t size, size_t alignment)
{
    if (size == 0)
    {
        size = alignment;
    }
    else
    {
        auto remainder = size % alignment;
        if (remainder > 0)
        {
            size += (alignment - remainder);
        }
    }
    return size;
}

size_t nnfusion::MemoryAllocator::cur_allocated() const
{
    return (prev(m_node_list.end())->m_state == block_state::FREE)
               ? numeric_limits<size_t>::max() - prev(m_node_list.end())->m_size
               : numeric_limits<size_t>::max();
}

size_t nnfusion::MemoryAllocator::memory_in_use() const
{
    size_t allocated = 0;
    for (const node& n : m_node_list)
    {
        if (n.m_state == block_state::ALLOCATED)
            allocated += n.m_size;
    }
    return allocated;
}
LanguageUnit_p nnfusion::HostMemoryAllocator::emit_memory_alloc()
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_alloc"));
    if (FLAGS_fcustomized_mem_imp)
        return _lu;

    auto& lu = *_lu;
    if (m_max_allocated > 0)
    {
        if (!FLAGS_ffunction_codegen)
            lu << this->get_name() << "_memory_pool = (char *)malloc(" << m_max_allocated << ");\n";
        for (auto tensor : m_allocated_tensors)
        {
            NNFUSION_CHECK(tensor->get_pool() == this->get_name());
            if (tensor->get_pool_offset() == SIZE_MAX)
            {
                NNFUSION_LOG(NNFUSION_WARNING)
                    << tensor->get_name()
                    << " may refer an external tensor, nnfusion omits its memory allocation.";
                lu << "// ";
            }
            lu << tensor->get_name() << " = ("
               << element::get_backend_cstring(tensor->get_element_type()) << "*)("
               << this->get_name() << "_memory_pool+" << tensor->get_pool_offset() << ");\n";
        }
    }
    return _lu;
}

LanguageUnit_p nnfusion::HostMemoryAllocator::emit_memory_free()
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_free"));
    if (FLAGS_fcustomized_mem_imp || FLAGS_ffunction_codegen)
        return _lu;

    auto& lu = *_lu;
    lu << "free(" << this->get_name() + "_memory_pool);\n";
    return _lu;
}

LanguageUnit_p nnfusion::HostMemoryAllocator::emit_memory_set(int value)
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_memset"));
    auto& lu = *_lu;
    lu << "memset(" << this->get_name() + "_memory_pool, " << value << ", " << m_max_allocated
       << ");\n";
    return _lu;
}

LanguageUnit_p nnfusion::HLSLMemoryAllocator::emit_memory_init()
{
    LanguageUnit_p _lu(new LanguageUnit("declaration::" + this->get_name() + "_init"));
    auto& lu = *_lu;
    if (m_max_allocated > 0)
    {
        if (FLAGS_fhlsl_codegen_type == "cpp")
        {
            lu << "void* " << this->get_name() << "_memory_pool;\n";
        }
        else if (FLAGS_fhlsl_codegen_type == "csharp")
        {
            lu << "static IntPtr " << this->get_name() << "_memory_pool;\n";
        }

        for (auto tensor : m_allocated_tensors)
        {
            if (FLAGS_fhlsl_codegen_type == "cpp")
            {
                lu << "void* " << tensor->get_name() << ";\n";
            }
            else if (FLAGS_fhlsl_codegen_type == "csharp")
            {
                lu << "static IntPtr " << tensor->get_name() << ";\n";
            }
        }
    }
    return _lu;
}

LanguageUnit_p nnfusion::HLSLMemoryAllocator::emit_memory_alloc()
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_alloc"));
    if (FLAGS_fcustomized_mem_imp)
        return _lu;

    auto& lu = *_lu;
    if (m_max_allocated > 0)
    {
        if (!FLAGS_ffunction_codegen)
            lu << this->get_name() << "_memory_pool = dxMemAlloc(" << m_max_allocated << ");\n";
        for (auto tensor : m_allocated_tensors)
        {
            if (tensor->get_shared_tensor())
            {
                // this tensor can be shared with the grpah_0's tensor
                lu << tensor->get_name()
                   << " = graph_0::" << tensor->get_shared_tensor()->get_name() << ";\n";
                continue;
            }
            NNFUSION_CHECK(tensor->get_pool() == this->get_name());
            if (tensor->get_pool_offset() == SIZE_MAX)
            {
                NNFUSION_LOG(NNFUSION_WARNING)
                    << tensor->get_name()
                    << " may refer an external tensor, nnfusion omits its memory allocation.";
                lu << "// ";
            }
            if (FLAGS_fhlsl_codegen_type == "cpp")
            {
                lu << tensor->get_name() << " = (char*)" << this->get_name() << "_memory_pool + "
                   << tensor->get_pool_offset() << ";\n";
            }
            else if (FLAGS_fhlsl_codegen_type == "csharp")
            {
                lu << tensor->get_name() << " = IntPtr.Add(" << this->get_name() << "_memory_pool, "
                   << tensor->get_pool_offset() << ");\n";
            }
        }
    }
    return _lu;
}

LanguageUnit_p nnfusion::HLSLMemoryAllocator::emit_memory_pool_offset(size_t offset)
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_offset"));
    auto& lu = *_lu;
    if (FLAGS_fhlsl_codegen_type == "cpp")
        lu << this->get_name() << "_memory_pool = (char*)(workspace + " << offset << ");\n";
    else if (FLAGS_fhlsl_codegen_type == "csharp")
        lu << this->get_name() << "_memory_pool = IntPtr.Add(workspace, " << offset << ");\n";
    return _lu;
}

LanguageUnit_p nnfusion::HLSLMemoryAllocator::emit_memory_free()
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_free"));
    if (FLAGS_fcustomized_mem_imp || FLAGS_ffunction_codegen)
        return _lu;

    auto& lu = *_lu;
    lu << "dxMemFree(" << this->get_name() + "_memory_pool);\n";
    return _lu;
}

LanguageUnit_p nnfusion::HLSLMemoryAllocator::emit_memory_set(int value)
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_memset"));
    return _lu;
}

nnfusion::MemoryAllocatorFactory::MemoryAllocatorFactory(size_t alignment, bool disable_reuse)
    : m_alignment(alignment)
    , m_disable_reuse(disable_reuse)

{
    NNFUSION_CHECK_WITH_EXCEPTION(m_alignment > 0, errors::InvalidArgument)
        << "Memory alignment must be > 0";
}

MemoryAllocator*
    nnfusion::MemoryAllocatorFactory::get_allocator(shared_ptr<descriptor::Tensor> tensor)
{
    auto group = tensor->get_group();
    NNFUSION_CHECK(group != "");
    std::string search_name = tensor->get_device_name() + "_" + group;
    if (tensor->is_memset())
    {
        NNFUSION_CHECK(tensor->get_memset_value() == 0) << "Tensor memset value must be 0";
        search_name += "_memset0";
    }
    if (m_allocator_list.find(search_name) != m_allocator_list.end())
    {
        return m_allocator_list[search_name];
    }
    else
    {
        if (tensor->is_memset())
        {
            // TODO(lingm): set disable_reuse = true after fixing no_reuse bad free issue
            if (tensor->is_RDMA_tensor())
            {
                auto device_type = tensor->get_device_type();
                RDMAMemoryAllocator* allocator =
                    new RDMAMemoryAllocator(m_alignment,
                                            m_disable_reuse,
                                            device_type,
                                            tensor->get_device_id(),
                                            "group_" + group + "_RDMA" + "_memset0");
                m_allocator_list[search_name] = allocator;
                return allocator;
            }
            else
            {
                MemoryAllocator* allocator = nullptr;
                switch (tensor->get_device_type())
                {
                case CUDA_GPU:
                {
                    allocator = new CUDAMemoryAllocator(m_alignment,
                                                        m_disable_reuse,
                                                        CUDA_GPU,
                                                        tensor->get_device_id(),
                                                        "group_" + group + "_memset0");
                    break;
                }
                case ROCM_GPU:
                {
                    allocator = new RocmMemoryAllocator(m_alignment,
                                                        m_disable_reuse,
                                                        ROCM_GPU,
                                                        tensor->get_device_id(),
                                                        "group_" + group + "_memset0");
                    break;
                }
                case GENERIC_CPU:
                {
                    allocator = new HostMemoryAllocator(m_alignment,
                                                        m_disable_reuse,
                                                        GENERIC_CPU,
                                                        tensor->get_device_id(),
                                                        "group_" + group + "_memset0");
                    break;
                }
                default: NNFUSION_LOG(ERROR) << "No valid allocator found: " << search_name; break;
                }
                if (allocator != nullptr)
                    m_allocator_list[search_name] = allocator;
                return allocator;
            }
        }
        else
        {
            if (tensor->is_RDMA_tensor())
            {
                auto device_type = tensor->get_device_type();
                RDMAMemoryAllocator* allocator =
                    new RDMAMemoryAllocator(m_alignment,
                                            m_disable_reuse,
                                            device_type,
                                            tensor->get_device_id(),
                                            "group_" + group + "_RDMA");
                m_allocator_list[search_name] = allocator;
                return allocator;
            }
            else
            {
                MemoryAllocator* allocator = nullptr;
                switch (tensor->get_device_type())
                {
                case CUDA_GPU:
                {
                    allocator = new CUDAMemoryAllocator(m_alignment,
                                                        m_disable_reuse,
                                                        CUDA_GPU,
                                                        tensor->get_device_id(),
                                                        "group_" + group);
                    break;
                }
                case ROCM_GPU:
                {
                    allocator = new RocmMemoryAllocator(m_alignment,
                                                        m_disable_reuse,
                                                        ROCM_GPU,
                                                        tensor->get_device_id(),
                                                        "group_" + group);
                    break;
                }
                case GENERIC_CPU:
                {
                    allocator = new HostMemoryAllocator(m_alignment,
                                                        m_disable_reuse,
                                                        GENERIC_CPU,
                                                        tensor->get_device_id(),
                                                        "group_" + group);
                    break;
                }
                case HLSL:
                {
                    allocator = new HLSLMemoryAllocator(m_alignment,
                                                        m_disable_reuse,
                                                        HLSL,
                                                        tensor->get_device_id(),
                                                        "group_" + group);
                    break;
                }
                default: NNFUSION_LOG(ERROR) << "No valid allocator found: " << search_name; break;
                }
                if (allocator != nullptr)
                    m_allocator_list[search_name] = allocator;
                return allocator;
            }
        }
    }
}
