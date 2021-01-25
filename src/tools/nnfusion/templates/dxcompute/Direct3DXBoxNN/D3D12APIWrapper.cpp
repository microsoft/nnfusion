#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>
#include <cassert>
#include <unordered_map>
#include <vector>
#include <map>

#define _USE_GPU_TIMER_
// #define _USE_DXC_

#include "D3D12Util.h"
#include "D3D12APIWrapper.h"

//namespace {
    static bool _USE_DESCRIPTOR_HEAP_ = false;

    struct dx_buffer_t
    {
        size_t size;
        ComPtr<ID3D12Resource> handle;

        // Added state management code.
        D3D12_RESOURCE_STATES state;
        void StateTransition(ID3D12GraphicsCommandList* pCmdList, D3D12_RESOURCE_STATES dstState)
        {
            if (dstState != state)
            {
                D3D12_RESOURCE_BARRIER barrier;
                ZeroMemory(&barrier, sizeof(barrier));
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = handle.Get();
                barrier.Transition.StateBefore = state;
                barrier.Transition.StateAfter = dstState;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                pCmdList->ResourceBarrier(1, &barrier);
                state = dstState;
            }
            else if (dstState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            {
                //Add UAV barrier
                D3D12_RESOURCE_BARRIER barrier;
                ZeroMemory(&barrier, sizeof(barrier));
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.UAV.pResource = handle.Get();
                pCmdList->ResourceBarrier(1, &barrier);
            }
        }
    };

    struct dx_tensor_t
    {
        std::vector<size_t> shape;
        std::string name, dtype;

        size_t NumElements() {
            return std::accumulate(shape.begin(), shape.end(), (size_t)1L, std::multiplies<size_t>());
        }

        size_t TypeSize() {
            for (int i = (int)dtype.size() - 1; i >= 0; --i) {
                if (!isdigit(dtype[i])) {
                    int bits = std::atoi(dtype.c_str() + i + 1);
                    if (bits % 8 > 0)
                        throw std::runtime_error("Data type bitsize must align with 8-bit byte type.");
                    return bits / 8;
                }
            }
            throw std::runtime_error(("Invalid data type name: " + dtype).c_str());
        }
    };

    struct dx_shader_t
    {
        int block[3], thread[3];
        std::vector<dx_tensor_t> inputs, outputs;
        std::string source;
        CD3DX12_SHADER_BYTECODE bytecode;

        // Added D3D12 resource ptr.
        ComPtr<ID3D12RootSignature> pRootSignature;
        ComPtr<ID3D12PipelineState> pPSO;
    };

    // Query heaps are used to allocate query objects.
    struct dx_query_heap_t
    {
        ComPtr<ID3D12QueryHeap> pHeap;
        ComPtr<ID3D12Resource> pReadbackBuffer;
        uint32_t curIdx;
        uint32_t totSize;
    };

    // Currently queries are only used to query GPU time-stamp.
    struct dx_query_t
    {
        uint32_t heapIdx;
        uint32_t queryIdxInHeap;
    };

    // Stream is wrapper of resources for record and execute commands.
    // Currently it only wraps commandlist, allocator and descriptor heaps.
    // Since all streams essentially will be submitted to a single DIRECT queue, their execution are not overlapped on GPU.
    // In the future we may submit streams to multiple queues for overlapped execution.
    struct dx_stream_t
    {
        // Set and get by device.
        uint64_t fenceVal = 0;
        enum class State
        {
            INRECORD,
            SUBMITTED,
        };

        // A stream is a wrapper of cmdlist, cmdallocator and descriptor heap.
        ComPtr<ID3D12GraphicsCommandList> pCmdList;
        ComPtr<ID3D12CommandAllocator> pCmdAllocator;
        ComPtr<ID3D12DescriptorHeap> pDescHeap;

        State state;
        uint32_t descIdxOffset = 0;

        void Reset()
        {
            pCmdAllocator->Reset();
            pCmdList->Reset(pCmdAllocator.Get(), nullptr);
            descIdxOffset = 0;
            state = State::INRECORD;
            if (_USE_DESCRIPTOR_HEAP_)
            {
                ID3D12DescriptorHeap* pDescHeaps[] = { pDescHeap.Get() };
                pCmdList->SetDescriptorHeaps(1, pDescHeaps);
            }
            queryHeapsNeedToResolve.clear();
        }

        std::vector<size_t> queryHeapsNeedToResolve;
    };

#ifdef _DEBUG
    static antares::D3DDevice device(true, true);
#else
    static antares::D3DDevice device(false, false);
#endif

    // Use unique_ptr to ensure the D3D resources are released when app exits.
    // TODO: Release buffers in somewhere to avoid running out GPU memory.
    static std::vector<std::unique_ptr<dx_buffer_t>> buffers;

    // Allocate individual queries from heaps for higher efficiency.
    // Since they consume little memory, we can release heaps when app exits.
    static std::vector<dx_query_heap_t> globalQueryHeaps;

    // Reuse queries since they are small objects and may be frequently created.
    // Use unique_ptr to grantee it will be released when app exits.
    static std::vector<std::unique_ptr<dx_query_t>> globalFreeQueries;

    static std::string get_between(const std::string& source, const std::string& begin, const std::string& end, const char* def = "")
    {
        std::string ret;
        int idx = (int)source.find(begin);
        if (idx < 0)
            return def;
        idx += (int)begin.size();
        int tail = (int)source.find(end, idx);
        if (idx < 0)
            return def;
        return source.substr(idx, tail - idx);
    }

    static void* defaultStream = nullptr;

    static std::map<void*, void*> memBlocks;

    static std::map<void*, void*>::const_iterator map_device_ptr(void* vPtr)
    {
        assert(memBlocks.empty() == false);
        auto nextIter = memBlocks.upper_bound(vPtr);
        assert(nextIter != memBlocks.begin());
        auto iter = std::prev(nextIter);
        assert(static_cast<char*>(vPtr) - static_cast<char*>(iter->first) >= 0);
        return iter;
    }


int dxInit(int flags)
{
    if (!defaultStream)
    {
        device.Init();
        _USE_DESCRIPTOR_HEAP_ = flags;
        if (_USE_DESCRIPTOR_HEAP_)
            fprintf(stderr, "[INFO] D3D12: Descriptor heap is enabled.\n\n");
        else
            fprintf(stderr, "[INFO] D3D12: Descriptor heap is disabled.\n\n");
        defaultStream = (void*)1LU;
        defaultStream = dxStreamCreate();
    }
    return 0;
}

void* dxMemAlloc(size_t bytes)
{
    if (dxInit(0) != 0)
        return nullptr;

    std::unique_ptr<dx_buffer_t> buff = std::make_unique<dx_buffer_t>();
    buff->size = bytes;
    device.CreateGPUOnlyResource(bytes, &buff->handle);
    assert(buff->handle.Get() != nullptr);
    buff->state = D3D12_RESOURCE_STATE_COMMON;

    buffers.push_back(std::move(buff));
    void* devicePtr = buffers.back().get();

    void* virtualPtr = VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
    assert(virtualPtr != nullptr);

    memBlocks[virtualPtr] = devicePtr;
    return virtualPtr;
}

int dxMemFree(void* vPtr)
{
    VirtualFree(vPtr, 0, MEM_RELEASE);
    memBlocks.erase(vPtr);
    return 0;
}

int dxShaderGetProperty(void* hShader, int arg_index, size_t* num_elements, size_t* type_size, const char** dtype_name)
{
    auto hd = (dx_shader_t*)hShader;
    size_t count, tsize;
    std::string dtype;
    if (arg_index < hd->inputs.size())
    {
        count = hd->inputs[arg_index].NumElements();
        dtype = hd->inputs[arg_index].dtype;
        tsize = hd->inputs[arg_index].TypeSize();
    }
    else
    {
        count = hd->outputs[arg_index - hd->inputs.size()].NumElements();
        dtype = hd->outputs[arg_index - hd->inputs.size()].dtype;
        tsize = hd->outputs[arg_index - hd->inputs.size()].TypeSize();
    }
    if (num_elements != nullptr)
        *num_elements = count;
    if (type_size != nullptr)
        *type_size = tsize;
    if (dtype_name != nullptr)
    {
        static char lastDtypeName[MAX_PATH];
        strncpy_s(lastDtypeName, dtype.c_str(), sizeof(lastDtypeName));
        *dtype_name = lastDtypeName;
    }
    return 0;
}

void* dxShaderLoad(const char* src, int* num_inputs, int* num_outputs)
{
    if (dxInit(0) != 0)
        return nullptr;

    std::string source = src;
    const char proto[] = "file://";
    if (strncmp(source.c_str(), proto, sizeof(proto) - 1) == 0) {
        std::ifstream t(src + sizeof(proto) - 1, ios_base::binary);
        if (t.fail())
            return nullptr;
        std::string _((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        source = std::move(_);
    }

    dx_shader_t* handle = new dx_shader_t;
    handle->source = source;

#ifdef _USE_DXC_
    // Use cs_6_0 since dxc only supports cs_6_0 or higher shader models.
    auto computeShader = antares::DXCompiler::Get()->Compile(source.data(), (uint32_t)source.size(), L"CSMain", L"cs_6_0");
    if (computeShader != nullptr)
        handle->bytecode = CD3DX12_SHADER_BYTECODE(computeShader->GetBufferPointer(), computeShader->GetBufferSize());
#else
    ComPtr<ID3DBlob> computeShader = nullptr;
    if (D3DCompile(source.data(), source.size(), NULL, NULL, NULL, "CSMain", "cs_5_1", 0, 0, &computeShader, NULL) >= 0 && computeShader != nullptr)
        handle->bytecode = CD3DX12_SHADER_BYTECODE(computeShader.Get());
#endif
    if (computeShader == nullptr) {
        //delete handle;
        return nullptr;
    }

    auto ssplit = [](const std::string& source, const std::string& delim) -> std::vector<std::string> {
        std::vector<std::string> ret;
        int it = 0, next;
        while (next = (int)source.find(delim, it), next >= 0) {
            ret.push_back(source.substr(it, next - it));
            it = next + (int)delim.size();
        }
        ret.push_back(source.substr(it));
        return std::move(ret);
    };

    auto parse_tensor = [&](const std::string& param) -> dx_tensor_t {
        dx_tensor_t ret;
        auto parts = ssplit(param, "/");
        for (auto it : ssplit(parts[0], "-"))
            ret.shape.push_back(std::atoi(it.c_str()));
        ret.dtype = parts[1];
        ret.name = parts[2];
        return ret;
    };

    auto str_params = get_between(source, "///", "\n");
    auto arr_params = ssplit(str_params, ":");
    assert(arr_params.size() == 2);
    auto in_params = ssplit(arr_params[0], ","), out_params = ssplit(arr_params[1], ",");
    if (in_params.size() == 1 && !in_params[0].size())
        in_params.clear();

    for (int i = 0; i < in_params.size(); ++i)
        handle->inputs.push_back(parse_tensor(in_params[i]));
    for (int i = 0; i < out_params.size(); ++i)
        handle->outputs.push_back(parse_tensor(out_params[i]));

    handle->block[0] = std::atoi(get_between(source, "// [thread_extent] blockIdx.x = ", "\n", "1").c_str());
    handle->block[1] = std::atoi(get_between(source, "// [thread_extent] blockIdx.y = ", "\n", "1").c_str());
    handle->block[2] = std::atoi(get_between(source, "// [thread_extent] blockIdx.z = ", "\n", "1").c_str());
    handle->thread[0] = std::atoi(get_between(source, "// [thread_extent] threadIdx.x = ", "\n", "1").c_str());
    handle->thread[1] = std::atoi(get_between(source, "// [thread_extent] threadIdx.y = ", "\n", "1").c_str());
    handle->thread[2] = std::atoi(get_between(source, "// [thread_extent] threadIdx.z = ", "\n", "1").c_str());

    assert(INT64(handle->thread[0]) * handle->thread[1] * handle->thread[2] <= 1024);
    if (num_inputs != nullptr)
        *num_inputs = (int)handle->inputs.size();
    if (num_outputs != nullptr)
        *num_outputs = (int)handle->outputs.size();

    // Added code to actually create D3D resources needed for shader launch.
    auto& hd = handle;

    ComPtr<ID3D12RootSignature>& m_computeRootSignature = hd->pRootSignature;
    ComPtr<ID3D12PipelineState>& m_computeState = hd->pPSO;
    D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc{};

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
    std::vector<CD3DX12_ROOT_PARAMETER1> computeRootParameters;
    if (_USE_DESCRIPTOR_HEAP_)
    {
        // Prepare Root
        computeRootParameters.resize(1);
        CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
        // D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE is needed to disable unproper driver optimization.
        ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, (uint32_t)hd->inputs.size(), 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, 0);
        ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, (uint32_t)hd->outputs.size(), 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, (uint32_t)hd->inputs.size());

        computeRootParameters[0].InitAsDescriptorTable(2, ranges);
        computeRootSignatureDesc.Init_1_1((UINT)computeRootParameters.size(),
            computeRootParameters.data());
    }
    else
    {
        // Prepare Root
        computeRootParameters.resize(hd->inputs.size() + hd->outputs.size());
        for (int i = 0; i < hd->inputs.size(); ++i)
            computeRootParameters[i].InitAsShaderResourceView(i);
        for (int i = 0; i < hd->outputs.size(); ++i)
            computeRootParameters[hd->inputs.size() + i].InitAsUnorderedAccessView(i);
        computeRootSignatureDesc.Init_1_1((UINT)computeRootParameters.size(), computeRootParameters.data());
    }

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;

    IFE(D3DX12SerializeVersionedRootSignature(&computeRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error));
    IFE(device.pDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_GRAPHICS_PPV_ARGS(m_computeRootSignature.ReleaseAndGetAddressOf())));

    computePsoDesc.CS = hd->bytecode;
    computePsoDesc.pRootSignature = m_computeRootSignature.Get();
    IFE(device.pDevice->CreateComputePipelineState(&computePsoDesc, IID_GRAPHICS_PPV_ARGS(m_computeState.ReleaseAndGetAddressOf())));

    return handle;
}

int dxShaderUnload(void* hShader)
{
    if (hShader != nullptr)
        delete (dx_shader_t*)hShader;
    return 0;
}

void* dxStreamCreate()
{
    if (dxInit(0) != 0)
        return nullptr;

    dx_stream_t* pStream = new dx_stream_t;

    // Create 
    IFE(device.pDevice->CreateCommandAllocator(device.CommandListType, IID_GRAPHICS_PPV_ARGS(pStream->pCmdAllocator.ReleaseAndGetAddressOf())));
    IFE(device.pDevice->CreateCommandList(0, device.CommandListType, pStream->pCmdAllocator.Get(), nullptr, IID_GRAPHICS_PPV_ARGS(pStream->pCmdList.ReleaseAndGetAddressOf())));
    pStream->pCmdList->Close(); // Close it and then reset it with pStream->Reset().

    if (_USE_DESCRIPTOR_HEAP_)
    {
        // Create per-stream descriptor heap.
        // const UINT MAX_HEAP_SIZE = (1U << 20);
        // Resource binding tier1/2 devices and some of the tier3 devices (e.g. NVIDIA Turing GPUs) DO-NOT support descriptor heap size larger than 1000000.
        const UINT MAX_HEAP_SIZE = 65536;
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        memset(&desc, 0, sizeof(desc));
        ZeroMemory(&desc, sizeof(desc));
        desc.NumDescriptors = MAX_HEAP_SIZE;
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        IFE(device.pDevice->CreateDescriptorHeap(&desc, IID_GRAPHICS_PPV_ARGS(pStream->pDescHeap.ReleaseAndGetAddressOf())));
    }
    pStream->Reset();
    return pStream;
}

int dxStreamDestroy(void* hStream)
{
    if (hStream != nullptr)
        delete (dx_stream_t*)hStream;
    return 0;
}

int dxStreamSubmit(void* hStream)
{
    if (!hStream)
        hStream = defaultStream;

    auto pStream = (dx_stream_t*)hStream;
    if (pStream->state == dx_stream_t::State::INRECORD)
    {
        pStream->state = dx_stream_t::State::SUBMITTED;
        
        // Resolve all query heaps when necessary
        for (auto q : pStream->queryHeapsNeedToResolve)
        {
            // We just resolve full heap for simplicity.
            pStream->pCmdList->ResolveQueryData(globalQueryHeaps[q].pHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, globalQueryHeaps[q].totSize, globalQueryHeaps[q].pReadbackBuffer.Get(), 0);
        }
        // Submit
        pStream->pCmdList->Close();
        ID3D12CommandList* cmdlists[] = { pStream->pCmdList.Get() };
        device.pCommandQueue->ExecuteCommandLists(1, cmdlists);
       
        // Signal fence.
        pStream->fenceVal = device.SignalFence();
    }
    return 0;
}

int dxStreamSynchronize(void* hStream)
{
    if (!hStream)
        hStream = defaultStream;

    auto pStream = (dx_stream_t*)hStream;

    if (pStream->state == dx_stream_t::State::INRECORD)
    {
        dxStreamSubmit(hStream);
    }
    // Wait for fence value.
    device.WaitForFence(pStream->fenceVal);

    // Reset stream to record state
    pStream->Reset();
    return 0;
}


int dxMemcpyHtoDAsync(void* dst, void* src, size_t bytes, void *hStream)
{
    if (!hStream)
        hStream = defaultStream;

    // Currently work in synchronizing way to hide API differences
    int ret = dxStreamSynchronize(hStream);
    if (ret != 0)
        return ret;

    // TODO: reuse D3D resources and not to create new resources in every call.
    ComPtr<ID3D12Resource> deviceCPUSrcX;
    device.CreateUploadBuffer(bytes, &deviceCPUSrcX);

    // CPU copy
    device.MapAndCopyToResource(deviceCPUSrcX.Get(), src, bytes);

    // GPU copy
    auto deviceIter = map_device_ptr(dst);
    UINT64 offset = static_cast<char*>(dst) - static_cast<char*>(deviceIter->first);
    auto dst_buffer = (dx_buffer_t*)(deviceIter->second);
    ComPtr<ID3D12CommandAllocator> pCommandAllocator;
    ComPtr<ID3D12GraphicsCommandList> pCmdList;
    IFE(device.pDevice->CreateCommandAllocator(device.CommandListType, IID_GRAPHICS_PPV_ARGS(pCommandAllocator.ReleaseAndGetAddressOf())));
    IFE(device.pDevice->CreateCommandList(0, device.CommandListType, pCommandAllocator.Get(), nullptr, IID_GRAPHICS_PPV_ARGS(pCmdList.ReleaseAndGetAddressOf())));
    dst_buffer->StateTransition(pCmdList.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
    pCmdList->CopyBufferRegion(dst_buffer->handle.Get(), offset, deviceCPUSrcX.Get(), 0, bytes);
    dst_buffer->StateTransition(pCmdList.Get(), D3D12_RESOURCE_STATE_COMMON);
    IFE(pCmdList->Close());

    // Conservatively ensure all things have been done, though currently not necessary.
    device.AwaitExecution();

    ID3D12CommandList* cmdlists[] = { pCmdList.Get() };
    device.pCommandQueue->ExecuteCommandLists(1, cmdlists);
    device.AwaitExecution();
    return 0;
}

int dxMemcpyDtoHAsync(void* dst, void* src, size_t bytes, void* hStream)
{
    if (!hStream)
        hStream = defaultStream;

    // Currently work in synchronizing way to hide API differences
    int ret = dxStreamSynchronize(hStream);
    if (ret != 0)
        return ret;

    // Conservatively ensure all things have been done, though currently not necessary.
    device.AwaitExecution();

    ComPtr<ID3D12Resource> deviceCPUSrcX;
    device.CreateReadbackBuffer(bytes, &deviceCPUSrcX);

    // GPU copy
    auto deviceIter = map_device_ptr(src);
    UINT64 offset = static_cast<char*>(src) - static_cast<char*>(deviceIter->first);
    auto src_buffer = (dx_buffer_t*)(deviceIter->second);
    ComPtr<ID3D12CommandAllocator> pCommandAllocator;
    ComPtr<ID3D12GraphicsCommandList> pCmdList;
    IFE(device.pDevice->CreateCommandAllocator(device.CommandListType, IID_GRAPHICS_PPV_ARGS(pCommandAllocator.ReleaseAndGetAddressOf())));
    IFE(device.pDevice->CreateCommandList(0, device.CommandListType, pCommandAllocator.Get(), nullptr, IID_GRAPHICS_PPV_ARGS(pCmdList.ReleaseAndGetAddressOf())));
    src_buffer->StateTransition(pCmdList.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
    pCmdList->CopyBufferRegion(deviceCPUSrcX.Get(), 0, src_buffer->handle.Get(), offset, bytes);
    src_buffer->StateTransition(pCmdList.Get(), D3D12_RESOURCE_STATE_COMMON);
    IFE(pCmdList->Close());
    ID3D12CommandList* cmdlists[] = { pCmdList.Get() };
    device.pCommandQueue->ExecuteCommandLists(1, cmdlists);
    device.AwaitExecution();

    // CPU copy
    device.MapCopyFromResource(deviceCPUSrcX.Get(), dst, bytes);
    return 0;
}

int dxShaderLaunchAsync(void* hShader, void** buffers, void* hStream)
{
    if (!hStream)
        hStream = defaultStream;

    auto hd = (dx_shader_t*)hShader;
    auto pStream = (dx_stream_t*)hStream;
    assert(pStream->state == dx_stream_t::State::INRECORD);

    std::vector<void*> devicePtrs;
    std::vector<UINT64> offsets;
    devicePtrs.reserve(hd->inputs.size() + hd->outputs.size());
    offsets.reserve(hd->inputs.size() + hd->outputs.size());
    for (int i = 0; i < hd->inputs.size(); ++i)
    {
        auto deviceIter = map_device_ptr(buffers[i]);
        devicePtrs.push_back(deviceIter->second);
        offsets.push_back(static_cast<char*>(buffers[i]) - static_cast<char*>(deviceIter->first));
    }
    for (int i = 0; i < hd->outputs.size(); ++i)
    {
        auto deviceIter = map_device_ptr(buffers[hd->inputs.size() + i]);
        devicePtrs.push_back(deviceIter->second);
        offsets.push_back(static_cast<char*>(buffers[hd->inputs.size() + i]) - static_cast<char*>(deviceIter->first));
    }

    // Handle state transition.
    for (int i = 0; i < hd->inputs.size(); ++i)
    {
        ((dx_buffer_t*)devicePtrs[i])->StateTransition(pStream->pCmdList.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    for (int i = 0; i < hd->outputs.size(); ++i)
    {
        ((dx_buffer_t*)devicePtrs[hd->inputs.size() + i])->StateTransition(pStream->pCmdList.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    pStream->pCmdList->SetComputeRootSignature(hd->pRootSignature.Get());
    pStream->pCmdList->SetPipelineState(hd->pPSO.Get());


    if (_USE_DESCRIPTOR_HEAP_)
    {
        auto handleCPU = pStream->pDescHeap->GetCPUDescriptorHandleForHeapStart();
        auto handleGPU = pStream->pDescHeap->GetGPUDescriptorHandleForHeapStart();
        auto nStep = device.pDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        handleCPU.ptr += pStream->descIdxOffset * nStep;
        handleGPU.ptr += pStream->descIdxOffset * nStep;
        pStream->descIdxOffset += (uint32_t)hd->inputs.size() + (uint32_t)hd->outputs.size();

        // Create SRV and UAVs at shader launch time.
        // A higher performance solution may be pre-create it in CPU desc heaps and then copy the desc to GPU heaps in realtime.
        for (size_t i = 0; i < hd->inputs.size(); ++i)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
            ZeroMemory(&srvDesc, sizeof(srvDesc));
            srvDesc.Format = DXGI_FORMAT_UNKNOWN;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            assert(offsets[i] % (uint32_t)hd->inputs[i].TypeSize() == 0);
            srvDesc.Buffer.FirstElement = offsets[i] / (uint32_t)hd->inputs[i].TypeSize();
            srvDesc.Buffer.NumElements = (uint32_t)hd->inputs[i].NumElements();
            srvDesc.Buffer.StructureByteStride = (uint32_t)hd->inputs[i].TypeSize();
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

            device.pDevice->CreateShaderResourceView(((dx_buffer_t*)devicePtrs[i])->handle.Get(), &srvDesc, handleCPU);
            handleCPU.ptr += nStep;
        }
        for (size_t i = 0; i < hd->outputs.size(); ++i)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc;
            ZeroMemory(&uavDesc, sizeof(uavDesc));
            uavDesc.Format = DXGI_FORMAT_UNKNOWN;
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            assert(offsets[hd->inputs.size() + i] % (uint32_t)hd->outputs[i].TypeSize() == 0);
            uavDesc.Buffer.FirstElement = offsets[hd->inputs.size() + i] / (uint32_t)hd->outputs[i].TypeSize();
            uavDesc.Buffer.NumElements = (uint32_t)hd->outputs[i].NumElements();
            uavDesc.Buffer.StructureByteStride = (uint32_t)hd->outputs[i].TypeSize();
            device.pDevice->CreateUnorderedAccessView(((dx_buffer_t*)devicePtrs[hd->inputs.size() + i])->handle.Get(), nullptr, &uavDesc, handleCPU);
            handleCPU.ptr += nStep;
        }

        pStream->pCmdList->SetComputeRootDescriptorTable(0, handleGPU);
    }
    else
    {

        for (uint32_t i = 0; i < hd->inputs.size(); ++i)
            pStream->pCmdList->SetComputeRootShaderResourceView(i, ((dx_buffer_t*)devicePtrs[i])->handle.Get()->GetGPUVirtualAddress() + offsets[i]);
        for (uint32_t i = 0; i < hd->outputs.size(); ++i)
            pStream->pCmdList->SetComputeRootUnorderedAccessView((UINT)hd->inputs.size() + i, ((dx_buffer_t*)devicePtrs[hd->inputs.size() + i])->handle.Get()->GetGPUVirtualAddress() + offsets[hd->inputs.size() + i]);
    }

#ifdef _USE_GPU_TIMER_
    int m_nTimerIndex = device.AllocTimerIndex();
    // Set StartTimer here to only consider kernel execution time.
    device.StartTimer(pStream->pCmdList.Get(), m_nTimerIndex);
#endif
    pStream->pCmdList->Dispatch(hd->block[0], hd->block[1], hd->block[2]);
#ifdef _USE_GPU_TIMER_
    device.StopTimer(pStream->pCmdList.Get(), m_nTimerIndex);
#endif
    return 0;
}

void* dxEventCreate()
{
    if (dxInit(0) != 0)
        return nullptr;

    // Return available query slots.
    if (globalFreeQueries.size() > 0)
    {
        auto ret = globalFreeQueries.back().release();
        globalFreeQueries.pop_back();
        return ret;
    }

    // If no free heaps, create new heap
    if (globalQueryHeaps.size() == 0 ||
        globalQueryHeaps.back().curIdx >= globalQueryHeaps.back().totSize)
    {
        dx_query_heap_t qheap;
        const UINT MAX_QUERY_NUM = 1024;

        D3D12_HEAP_PROPERTIES HeapProps;
        HeapProps.Type = D3D12_HEAP_TYPE_READBACK;
        HeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        HeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        HeapProps.CreationNodeMask = 1;
        HeapProps.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC BufferDesc;
        BufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        BufferDesc.Alignment = 0;
        BufferDesc.Width = sizeof(uint64_t) * MAX_QUERY_NUM;
        BufferDesc.Height = 1;
        BufferDesc.DepthOrArraySize = 1;
        BufferDesc.MipLevels = 1;
        BufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        BufferDesc.SampleDesc.Count = 1;
        BufferDesc.SampleDesc.Quality = 0;
        BufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        BufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        IFE(device.pDevice->CreateCommittedResource(&HeapProps, D3D12_HEAP_FLAG_NONE, &BufferDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_GRAPHICS_PPV_ARGS(qheap.pReadbackBuffer.ReleaseAndGetAddressOf())));

        D3D12_QUERY_HEAP_DESC QueryHeapDesc;
        QueryHeapDesc.Count = MAX_QUERY_NUM;
        QueryHeapDesc.NodeMask = 1;
        QueryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        IFE(device.pDevice->CreateQueryHeap(&QueryHeapDesc, IID_GRAPHICS_PPV_ARGS(qheap.pHeap.ReleaseAndGetAddressOf())));

        qheap.curIdx = 0;
        qheap.totSize = MAX_QUERY_NUM;
        globalQueryHeaps.push_back(qheap);
    }

    // Assume heap has free slots. 
    dx_query_t* ret = new dx_query_t;
    ret->heapIdx = (uint32_t)globalQueryHeaps.size() - 1;
    ret->queryIdxInHeap = globalQueryHeaps.back().curIdx;
    globalQueryHeaps.back().curIdx++;
    return ret;
}

int dxEventDestroy(void* hEvent)
{
    if (hEvent == nullptr)
        return -1;

    // We just push queries for reuse.
    // Since queries only consume little memory, we only actually release them when app exits.
    std::unique_ptr<dx_query_t> q((dx_query_t*)hEvent);
    globalFreeQueries.push_back(std::move(q));
    return 0;
}

int dxEventRecord(void* hEvent, void* hStream)
{
    if (!hStream)
        hStream = defaultStream;

    auto pQuery = (dx_query_t*)hEvent;
    auto pStream = (dx_stream_t*)hStream;
    // Record commandlist.
    pStream->pCmdList->EndQuery(
        globalQueryHeaps[pQuery->heapIdx].pHeap.Get(),
        D3D12_QUERY_TYPE_TIMESTAMP, pQuery->queryIdxInHeap);

    // Also record the heaps needed to resolve.
    // Since there are only few number of heaps (in most cases, just 1), we use a linear search.
    for (auto q : pStream->queryHeapsNeedToResolve)
    {
        if (q == pQuery->heapIdx)
            return 0;
    }
    pStream->queryHeapsNeedToResolve.push_back(pQuery->heapIdx);
    return 0;
}

float dxEventElapsedTime(void* hStart, void* hStop)
{
    auto pQueryStart = (dx_query_t*)hStart;
    auto pQueryEnd = (dx_query_t*)hStop;

    // Map readback buffer and read out data, assume the query heaps have already been resolved.
    uint64_t* pData;
    uint64_t timeStampStart = 0;
    uint64_t timeStampEnd = 0;
    IFE(globalQueryHeaps[pQueryStart->heapIdx].pReadbackBuffer->Map(0, nullptr, reinterpret_cast<void**>(&pData)));
    timeStampStart = pData[pQueryStart->queryIdxInHeap];

    if (pQueryEnd->heapIdx == pQueryStart->heapIdx)
    {
        // If in same heap, just read out end data.
        timeStampEnd = pData[pQueryEnd->queryIdxInHeap];
    }
    else
    {
        // Otherwise, map heap and read.
        uint64_t* pDataEnd;
        IFE(globalQueryHeaps[pQueryEnd->heapIdx].pReadbackBuffer->Map(0, nullptr, reinterpret_cast<void**>(&pDataEnd)));
        timeStampEnd = pDataEnd[pQueryEnd->queryIdxInHeap];
        globalQueryHeaps[pQueryEnd->heapIdx].pReadbackBuffer->Unmap(0, nullptr);
    }
    globalQueryHeaps[pQueryStart->heapIdx].pReadbackBuffer->Unmap(0, nullptr);

    uint64_t GpuFrequency;
    IFE(device.pCommandQueue->GetTimestampFrequency(&GpuFrequency));
    return static_cast<float>(timeStampEnd - timeStampStart) / static_cast<float>(GpuFrequency);
}
//}
