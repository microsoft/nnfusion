// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>
#include <cassert>
#include <unordered_map>
#include <vector>
#include <map>
#include <locale>
#include <codecvt>

#define _USE_GPU_TIMER_
#define _USE_DXC_

#define ANTARES_EXPORTS

#include "D3D12Util.h"
#include "D3D12APIWrapper.h"

#if _DEBUG
#define DEBUG_PRINT(msg) (fprintf(stderr, "[DEBUG] %s\n", msg), fflush(stderr))
#else
#define DEBUG_PRINT(msg)
#endif

namespace base64 {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    typedef unsigned char BYTE;

    static inline bool is_base64(BYTE c) {
        return (isalnum(c) || (c == '+') || (c == '/'));
    }

    std::string encode(BYTE* buf, unsigned int bufLen) {
        std::string ret;
        int i = 0;
        int j = 0;
        BYTE char_array_3[3];
        BYTE char_array_4[4];

        while (bufLen--) {
            char_array_3[i++] = *(buf++);
            if (i == 3) {
                char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
                char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
                char_array_4[3] = char_array_3[2] & 0x3f;

                for (i = 0; (i < 4); i++)
                    ret += base64_chars[char_array_4[i]];
                i = 0;
            }
        }

        if (i)
        {
            for (j = i; j < 3; j++)
                char_array_3[j] = '\0';

            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (j = 0; (j < i + 1); j++)
                ret += base64_chars[char_array_4[j]];

            while ((i++ < 3))
                ret += '=';
        }

        return ret;
    }

    std::vector<BYTE> decode(std::string const& encoded_string) {
        int in_len = encoded_string.size();
        int i = 0;
        int j = 0;
        int in_ = 0;
        BYTE char_array_4[4], char_array_3[3];
        std::vector<BYTE> ret;

        while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
            char_array_4[i++] = encoded_string[in_]; in_++;
            if (i == 4) {
                for (i = 0; i < 4; i++)
                    char_array_4[i] = base64_chars.find(char_array_4[i]);

                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

                for (i = 0; (i < 3); i++)
                    ret.push_back(char_array_3[i]);
                i = 0;
            }
        }

        if (i) {
            for (j = i; j < 4; j++)
                char_array_4[j] = 0;

            for (j = 0; j < 4; j++)
                char_array_4[j] = base64_chars.find(char_array_4[j]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
        }

        return ret;
    }
}

namespace {
    static int _USE_DESCRIPTOR_HEAP_ = 0;

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

    struct VectorHasher {
        int operator()(const std::vector<size_t>& V) const {
            int hash = V.size();
            for (auto& i : V) {
                hash ^= (i ^ (i >> 32)) + 0x9e3779b9L;
            }
            return hash;
        }
    };

    std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) {
        size_t start_pos = 0;
        while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
        }
        return str;
    }

    struct dx_shader_t
    {
        unsigned block[3];
        std::vector<int> cbuffer_sizes;
        std::vector<dx_tensor_t> inputs, outputs;
        std::string source;
        std::vector<base64::BYTE> bytecode;

        // Added D3D12 resource ptr.
        ComPtr<ID3D12PipelineState> pPSO_ht;
        ComPtr<ID3D12RootSignature> pRootSignature;
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


    static std::shared_ptr<antares::D3DDevice> device;

    static void* defaultStream = nullptr;

    static std::map<void*, void*> memBlocks;

    static std::map<void*, void*>::const_iterator map_device_ptr(void* vPtr)
    {
        assert(!memBlocks.empty());
        auto iter = memBlocks.lower_bound(vPtr);
        if (iter == memBlocks.end() || size_t(iter->first) > size_t(vPtr))
            --iter;
        return iter;
    }

    static std::vector<std::string> ssplit(const std::string& source, const std::string& delim, bool allow_empty = false) {
        std::vector<std::string> ret;
        int it = 0, next;
        while (next = (int)source.find(delim, it), next >= 0) {
            if (next > it || allow_empty)
                ret.push_back(source.substr(it, next - it));
            it = next + (int)delim.size();
        }
        if (it < source.size() || allow_empty)
            ret.push_back(source.substr(it));
        return std::move(ret);
    }

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
}

int dxInit(int flags = 1, int ord = 0)
{
    DEBUG_PRINT(__func__);

    if (device == nullptr) {
#ifdef _DEBUG
        device = std::make_shared<antares::D3DDevice>(true, true);
#else
        device = std::make_shared<antares::D3DDevice>(false, false);
#endif
        device->Init(ord);

        // flags = -1: MODE-0 for xbox and MODE-1 for others
        // flags = 0: MODE-0 - disable descriptor heap
        // flags = 1: MODE-1 - enable descriptor heap for concrete shape (by default)
        // flags = 2: MODE-2 - enable descriptor heap, with maximum address boundary setting
        if (flags == -1) {
#ifdef _GAMING_XBOX_SCARLETT
            flags = 0;
#else
            flags = 1;
#endif
        }
        _USE_DESCRIPTOR_HEAP_ = flags;

        if (defaultStream != nullptr)
            throw std::runtime_error("Unexpected initialization of defaultStream.");
        defaultStream = (void*)1LU;
        defaultStream = dxStreamCreate();
    }
    return 0;
}

int dxFinalize() {
    DEBUG_PRINT(__func__);

    device = nullptr;
    defaultStream = nullptr;
    return 0;
}

static std::unordered_map<size_t, std::vector<void*>> unused_buffers;
static std::unordered_map<void*, size_t> buffer_slots;

inline size_t compute_slotsize(size_t &value) {
    static const int tab32[32] = {
      0,  9,  1, 10, 13, 21,  2, 29, 11, 14, 16, 18, 22, 25,  3, 30,
      8, 12, 20, 28, 15, 17, 24,  7, 19, 27, 23,  6, 26,  5,  4, 31};

    value -= 1;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;

    size_t slot_id;
    if (value > (1LL << 30))
        slot_id = 32 + (value - (1LL << 30)) / (1LL << 30);
    else
        slot_id = tab32[(uint32_t)(value * 0x07C4ACDD) >> 27];
    value += 1;
    return slot_id;
}

void* dxMemAlloc(size_t bytes)
{
    DEBUG_PRINT(__func__);

    if (dxInit() != 0)
        return nullptr;

    auto slot_id = compute_slotsize(bytes);
    auto& slot = unused_buffers[slot_id];
    if (slot.size()) {
        void* buff = slot.back();
        slot.pop_back();
        return buff;
    }

    auto buff = new dx_buffer_t();
    buff->size = bytes;
    device->CreateGPUOnlyResource(bytes, &buff->handle);
    assert(buff->handle.Get() != nullptr);
    buff->state = D3D12_RESOURCE_STATE_COMMON;

    void* virtualPtr = VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
    assert(virtualPtr != nullptr);
    buffer_slots[virtualPtr] = slot_id;

    memBlocks[virtualPtr] = buff;
    return virtualPtr;
}

int dxMemFree(void* virtualPtr)
{
    DEBUG_PRINT(__func__);

    auto it = buffer_slots.find(virtualPtr);
    assert(it != buffer_slots.end());
    unused_buffers[it->second].push_back(virtualPtr);
    return 0;

    VirtualFree(virtualPtr, 0, MEM_RELEASE);
    memBlocks.erase(virtualPtr);
    return 0;
}

static std::wstring default_compat = L"cs_6_0";

int dxModuleSetCompat(const char* compat_name) {
    if (*compat_name == '*') {
#ifdef _GAMING_XBOX_SCARLETT
        ::default_compat = L"cs_6_6";
#else
        ::default_compat = L"cs_6_5";
#endif
        return 0;
    }

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    ::default_compat = converter.from_bytes(compat_name);
    return 0;
}

void* dxShaderLoad_v3(const char* shader_src)
{
    DEBUG_PRINT(__func__);

    if (dxInit() != 0)
        return nullptr;

    auto source = shader_src;

    dx_shader_t* handle = new dx_shader_t;
    handle->source = source;

    std::string str_params;
    std::vector<std::string> arr_params, in_params, out_params;

    {
        str_params = get_between(source, " -- ", "\n");
        arr_params = ssplit(str_params, " -> ", true);
        assert(arr_params.size() == 2);
        in_params = ssplit(arr_params[0] + ", ", "], ");
        out_params = ssplit(arr_params[1] + ", ", "], ");
    }
    if (!arr_params[0].size())
        in_params.clear();

    auto parse_tensor = [&](const std::string & param) -> dx_tensor_t {
        dx_tensor_t ret;
        auto parts = ssplit(param, ":");
        ret.name = parts[0];
        parts = ssplit(parts[1], "[");
        ret.dtype = parts[0];
        for (auto it : ssplit(parts[1], ", "))
            ret.shape.push_back(std::atoi(it.c_str()));
        return ret;
    };

    for (int i = 0; i < in_params.size(); ++i)
        handle->inputs.push_back(parse_tensor(in_params[i]));
    for (int i = 0; i < out_params.size(); ++i)
        handle->outputs.push_back(parse_tensor(out_params[i]));

    auto sections = ssplit(get_between(source, "\n\n@@PACK:", "\n"), "@@", true); // bx, by, bz, vamap, binary

    handle->block[0] = std::atoll(sections[0].c_str());
    handle->block[1] = std::atoll(sections[1].c_str());
    handle->block[2] = std::atoll(sections[2].c_str());
    // fprintf(stderr, "%lld %lld %lld [%s]\n", handle->block[0], handle->block[1], handle->block[2], sections[2].c_str());

    handle->cbuffer_sizes = {0};
    auto cbuffs = ssplit(sections[3], ",");
    for (auto it : cbuffs) {
        auto type = ssplit(it, "/")[0];
        if (type == "double" || type == "int64_t" || type == "llong")
            handle->cbuffer_sizes.push_back(2);
        else
            handle->cbuffer_sizes.push_back(1);
        handle->cbuffer_sizes[0] += handle->cbuffer_sizes.back();
    }

    assert(INT64(handle->thread[0]) * handle->thread[1] * handle->thread[2] <= 1024);

    // Added code to actually create D3D resources needed for shader launch.
    auto& hd = handle;

    ComPtr<ID3D12RootSignature>& m_computeRootSignature = hd->pRootSignature;

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
    std::vector<CD3DX12_ROOT_PARAMETER1> computeRootParameters;

    // Prepare Root
    CD3DX12_DESCRIPTOR_RANGE1 ranges[1];
    if (_USE_DESCRIPTOR_HEAP_)
    {
        computeRootParameters.resize(2);
        // D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE is needed to disable unproper driver optimization.
        ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, (uint32_t)hd->inputs.size() + hd->outputs.size(), 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, 0);
        computeRootParameters[0].InitAsDescriptorTable(1, ranges);

        if (hd->cbuffer_sizes[0] > 0) {
            computeRootParameters[1].InitAsConstants(hd->cbuffer_sizes[0], 0);
        } else {
            computeRootParameters.pop_back();
        }
    }
    else
    {
        computeRootParameters.resize(hd->inputs.size() + hd->outputs.size() + 1);
        for (int i = 0; i < hd->inputs.size(); ++i)
            computeRootParameters[i].InitAsUnorderedAccessView(i);
        for (int i = 0; i < hd->outputs.size(); ++i)
            computeRootParameters[hd->inputs.size() + i].InitAsUnorderedAccessView(hd->inputs.size() + i);

        if (hd->cbuffer_sizes[0] > 0)
            computeRootParameters[hd->inputs.size() + hd->outputs.size()].InitAsConstants(hd->cbuffer_sizes[0], 0);
        else
            computeRootParameters.pop_back();
    }
    computeRootSignatureDesc.Init_1_1((UINT)computeRootParameters.size(), computeRootParameters.data());

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;

    IFE(D3DX12SerializeVersionedRootSignature(&computeRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error));
    IFE(device->pDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_GRAPHICS_PPV_ARGS(m_computeRootSignature.ReleaseAndGetAddressOf())));

    hd->bytecode = base64::decode(sections[4]);
    return handle;
}

void dxShaderUnload(void* hShader)
{
    DEBUG_PRINT(__func__);

    free(hShader);
}

#define ACCEPT_VERSION "@VER__1|"

const char* dxModuleCompile(const char* module_src, size_t* out_size)
{
    DEBUG_PRINT(__func__);

    // Ensure code instead of file path
    std::string source = module_src;
    const char proto[] = "file://";
    if (strncmp(module_src, proto, sizeof(proto) - 1) == 0) {
        std::ifstream t(module_src + sizeof(proto) - 1, ios_base::binary);
        if (t.fail()) {
            fprintf(stderr, "[Error] Failed to read module file: %s\n", module_src + sizeof(proto) - 1);
            IFE(-1);
        }
        std::string _((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        source = std::move(_);
    }
    module_src = source.c_str();

    static std::string module_buff;
    module_buff = source;
    // Compile if code is not binary
    if (*module_src != '@') {
        source = ReplaceAll(source, "\r", ""), module_src = source.c_str();
        module_buff = ACCEPT_VERSION;
        int curr = 0, next = source.find("\n// --------", curr);
        while (~next) {
            next = source.find("\n\n", next + 1);
            auto meta = source.substr(curr, next - curr);
            module_buff += meta;
            int tail = source.find("\n// --------", next);
            auto body = source.substr(next, tail - next);

            size_t bx = std::atoi(get_between(body, "// [thread_extent] blockIdx.x = ", "\n", "1").c_str());
            size_t by = std::atoi(get_between(body, "// [thread_extent] blockIdx.y = ", "\n", "1").c_str());
            size_t bz = std::atoi(get_between(body, "// [thread_extent] blockIdx.z = ", "\n", "1").c_str());
            auto vamap = get_between(body, "// VAMAP: ", "\n", "");

#ifdef _USE_DXC_
            // Use cs_6_0 since dxc only supports cs_6_0 or higher shader models.
            auto computeShader = antares::DXCompiler::Get()->Compile(body.data(), (uint32_t)body.size(), L"CSMain", default_compat.c_str());
#else
            ComPtr<ID3DBlob> computeShader = nullptr, errMsg = nullptr;
            D3DCompile(source.data(), source.size(), NULL, NULL, NULL, "CSMain", "cs_5_1", 0, 0, &computeShader, &errMsg);
#endif
            if (computeShader == nullptr) {
                auto fname = get_between(meta, "// LOCAL: ", " -- ");
                fprintf(stderr, "[Error] Failed to compile shader function with name: %s\n", fname.c_str());
                IFE(-1);
            }

            module_buff += "\n\n@@PACK:" + std::to_string(bx) + "@@" + std::to_string(by) + "@@" + std::to_string(bz) + "@@" + vamap + "@@";
            module_buff += base64::encode((base64::BYTE*)(computeShader->GetBufferPointer()), computeShader->GetBufferSize());
            module_buff += "\n";

            curr = next = tail;
        }
    }

    if (out_size != nullptr)
        *out_size = module_buff.size();
    return (char*)module_buff.data();
}

void* dxModuleLoad(const char* module_src)
{
    DEBUG_PRINT(__func__);

    std::string out_buffer = dxModuleCompile(module_src, nullptr);
    module_src = out_buffer.data();
    // fprintf(stderr, "%s\n", module_src);

    if (strncmp(module_src, ACCEPT_VERSION, sizeof(ACCEPT_VERSION) - 1) != 0) {
        fprintf(stderr, "[Error] The version of compiled module isn't compatible.\n");
        IFE(-1);
    }

    auto& hShaderDict = *(new std::unordered_map<std::string, void*>);

    int curr = out_buffer.find("-------\n"), next;
    curr = out_buffer.find('\n', curr) + 1, next = out_buffer.find("\n\n", curr) + 2;

    auto kernel_slices = ssplit(module_src, "-------\n");
    for (int i = 1; i < kernel_slices.size(); ++i) {
        auto name = get_between(kernel_slices[i], "// LOCAL: ", " -- ");
        hShaderDict[name] = dxShaderLoad_v3(kernel_slices[i].c_str());
    }
    return &hShaderDict;
}

void dxModuleUnload(void* hModule)
{
    DEBUG_PRINT(__func__);

    auto& hShaderDict = *(std::unordered_map<std::string, void*>*)hModule;
    for (auto& it : hShaderDict)
        dxShaderUnload(it.second);
    delete& hShaderDict;
}

void* dxModuleGetShader(void* hModule, const char* fname)
{
    DEBUG_PRINT(__func__);

    auto& dict = *(std::unordered_map<std::string, void*>*)hModule;
    auto it = dict.find(fname);
    return it != dict.end() ? it->second : nullptr;
}

void* dxStreamCreate()
{
    DEBUG_PRINT(__func__);

    if (dxInit() != 0)
        return nullptr;

    dx_stream_t* pStream = new dx_stream_t;

    // Create 
    IFE(device->pDevice->CreateCommandAllocator(device->CommandListType, IID_GRAPHICS_PPV_ARGS(pStream->pCmdAllocator.ReleaseAndGetAddressOf())));
    IFE(device->pDevice->CreateCommandList(0, device->CommandListType, pStream->pCmdAllocator.Get(), nullptr, IID_GRAPHICS_PPV_ARGS(pStream->pCmdList.ReleaseAndGetAddressOf())));
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
        IFE(device->pDevice->CreateDescriptorHeap(&desc, IID_GRAPHICS_PPV_ARGS(pStream->pDescHeap.ReleaseAndGetAddressOf())));
    }
    pStream->Reset();
    return pStream;
}

int dxStreamDestroy(void* hStream)
{
    DEBUG_PRINT(__func__);

    if (hStream != nullptr)
        delete (dx_stream_t*)hStream;
    return 0;
}

int dxStreamSubmit(void* hStream)
{
    DEBUG_PRINT(__func__);

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
            pStream->pCmdList->ResolveQueryData(device->globalQueryHeaps[q].pHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, device->globalQueryHeaps[q].totSize, device->globalQueryHeaps[q].pReadbackBuffer.Get(), 0);
        }
        // Submit
        pStream->pCmdList->Close();
        ID3D12CommandList* cmdlists[] = { pStream->pCmdList.Get() };
        device->pCommandQueue->ExecuteCommandLists(1, cmdlists);
       
        // Signal fence.
        pStream->fenceVal = device->SignalFence();
    }
    return 0;
}

int dxStreamSynchronize(void* hStream)
{
    DEBUG_PRINT(__func__);

    if (!hStream)
        hStream = defaultStream;

    auto pStream = (dx_stream_t*)hStream;

    if (pStream->state == dx_stream_t::State::INRECORD)
    {
        dxStreamSubmit(hStream);
    }
    // Wait for fence value.
    device->WaitForFence(pStream->fenceVal);

    // Reset stream to record state
    pStream->Reset();
    return 0;
}

int dxMemcpyDtoDAsync(void* dst, void* src, size_t bytes, void* hStream)
{
    DEBUG_PRINT(__func__);

    if (!hStream)
        hStream = defaultStream;

    // GPU copy
    auto pStream = (dx_stream_t*)hStream;
    assert(pStream->state == dx_stream_t::State::INRECORD);

    auto deviceIter = map_device_ptr(dst), sourceIter = map_device_ptr(src);
    UINT64 offset = static_cast<char*>(dst) - static_cast<char*>(deviceIter->first);
    UINT64 offsetSrc = static_cast<char*>(src) - static_cast<char*>(sourceIter->first);
    auto dst_buffer = (dx_buffer_t*)(deviceIter->second), src_buffer = (dx_buffer_t*)(sourceIter->second);
    auto& pCmdList = pStream->pCmdList;

    dst_buffer->StateTransition(pCmdList.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
    pCmdList->CopyBufferRegion(dst_buffer->handle.Get(), offset, src_buffer->handle.Get(), offsetSrc, bytes);
    dst_buffer->StateTransition(pCmdList.Get(), D3D12_RESOURCE_STATE_COMMON);

    return 0;
}


int dxMemcpyHtoDAsync(void* dst, void* src, size_t bytes, void *hStream)
{
    DEBUG_PRINT(__func__);

    if (!hStream)
        hStream = defaultStream;

    // Currently work in synchronizing way to hide API differences
    int ret = dxStreamSynchronize(hStream);
    if (ret != 0)
        return ret;

    // TODO: reuse D3D resources and not to create new resources in every call.
    ComPtr<ID3D12Resource> deviceCPUSrcX;
    device->CreateUploadBuffer(bytes, &deviceCPUSrcX);

    // CPU copy
    device->MapAndCopyToResource(deviceCPUSrcX.Get(), src, bytes);

    // GPU copy
    auto pStream = (dx_stream_t*)hStream;
    assert(pStream->state == dx_stream_t::State::INRECORD);

    auto deviceIter = map_device_ptr(dst);
    UINT64 offset = static_cast<char*>(dst) - static_cast<char*>(deviceIter->first);
    auto dst_buffer = (dx_buffer_t*)(deviceIter->second);
    auto& pCmdList = pStream->pCmdList;

    dst_buffer->StateTransition(pCmdList.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
    pCmdList->CopyBufferRegion(dst_buffer->handle.Get(), offset, deviceCPUSrcX.Get(), 0, bytes);
    dst_buffer->StateTransition(pCmdList.Get(), D3D12_RESOURCE_STATE_COMMON);
    IFE(pCmdList->Close());

    // Conservatively ensure all things have been done, though currently not necessary.
    device->AwaitExecution();

    ID3D12CommandList* cmdlists[] = { pCmdList.Get() };
    device->pCommandQueue->ExecuteCommandLists(1, cmdlists);
    device->AwaitExecution();

    return dxStreamSynchronize(hStream);
}

void* dxMemHostRegister(void* dptr, unsigned int subres) {
    auto deviceIter = map_device_ptr(dptr);
    UINT64 offset = static_cast<char*>(dptr) - static_cast<char*>(deviceIter->first);
    auto src_buffer = (dx_buffer_t*)(deviceIter->second);
    void* result = nullptr;
    D3D12_RANGE range = { 0, static_cast<SIZE_T>(src_buffer->size) };
    src_buffer->handle->Map(subres, &range, &result);
    return ((char*)result) + offset;
}

void dxMemHostUnregister(void* dptr, unsigned int subres) {
    auto deviceIter = map_device_ptr(dptr);
    UINT64 offset = static_cast<char*>(dptr) - static_cast<char*>(deviceIter->first);
    auto src_buffer = (dx_buffer_t*)(deviceIter->second);
    void* result = nullptr;
    D3D12_RANGE range = { 0, static_cast<SIZE_T>(src_buffer->size) };
    src_buffer->handle->Unmap(subres, &range);
}

int dxMemcpyDtoHAsync(void* dst, void* src, size_t bytes, void* hStream)
{
    DEBUG_PRINT(__func__);

    if (!hStream)
        hStream = defaultStream;

    // Currently work in synchronizing way to hide API differences
    int ret = dxStreamSynchronize(hStream);
    if (ret != 0)
        return ret;

    // Conservatively ensure all things have been done, though currently not necessary.
    device->AwaitExecution();

    ComPtr<ID3D12Resource> deviceCPUSrcX;
    device->CreateReadbackBuffer(bytes, &deviceCPUSrcX);

    // GPU copy
    auto pStream = (dx_stream_t*)hStream;
    assert(pStream->state == dx_stream_t::State::INRECORD);

    auto deviceIter = map_device_ptr(src);
    UINT64 offset = static_cast<char*>(src) - static_cast<char*>(deviceIter->first);
    auto src_buffer = (dx_buffer_t*)(deviceIter->second);

    auto& pCmdList = pStream->pCmdList;
    src_buffer->StateTransition(pCmdList.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
    pCmdList->CopyBufferRegion(deviceCPUSrcX.Get(), 0, src_buffer->handle.Get(), offset, bytes);
    src_buffer->StateTransition(pCmdList.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    IFE(pCmdList->Close());
    ID3D12CommandList* cmdlists[] = { pCmdList.Get() };
    device->pCommandQueue->ExecuteCommandLists(1, cmdlists);
    device->AwaitExecution();

    // CPU copy
    device->MapCopyFromResource(deviceCPUSrcX.Get(), dst, bytes);
    return dxStreamSynchronize(hStream);
}

int dxShaderLaunchAsyncExt(void* hShader, void** buffers, int blocks, void* hStream)
{
    DEBUG_PRINT(__func__);

    if (!hStream)
        hStream = defaultStream;
    auto hd = (dx_shader_t*)hShader;
    auto pStream = (dx_stream_t*)hStream;
    assert(pStream->state == dx_stream_t::State::INRECORD);

    if (hd->pPSO_ht == nullptr) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc{};
        CD3DX12_SHADER_BYTECODE bytecode(hd->bytecode.data(), hd->bytecode.size());
        computePsoDesc.CS = bytecode;
        computePsoDesc.pRootSignature = hd->pRootSignature.Get();
        IFE(device->pDevice->CreateComputePipelineState(&computePsoDesc, IID_GRAPHICS_PPV_ARGS(hd->pPSO_ht.ReleaseAndGetAddressOf())));
    }

    std::vector<int> pargs;
    pargs.reserve(hd->cbuffer_sizes[0]);
    for (int i = 1, j = hd->inputs.size() + hd->outputs.size(); i < hd->cbuffer_sizes.size(); ++i, ++j) {
        auto regs = (int64_t)buffers[j];
        if (hd->cbuffer_sizes[i] == 2) {
            pargs.push_back(regs);
            pargs.push_back(regs >> 32);
        } else {
            pargs.push_back(regs);
        }
    }
    assert(pargs.size() == hd->cbuffer_sizes[0]);

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
        ((dx_buffer_t*)devicePtrs[i])->StateTransition(pStream->pCmdList.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    for (int i = 0; i < hd->outputs.size(); ++i)
    {
        ((dx_buffer_t*)devicePtrs[hd->inputs.size() + i])->StateTransition(pStream->pCmdList.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    pStream->pCmdList->SetComputeRootSignature(hd->pRootSignature.Get());
    pStream->pCmdList->SetPipelineState(hd->pPSO_ht.Get());

    if (_USE_DESCRIPTOR_HEAP_)
    {
        auto handleCPU = pStream->pDescHeap->GetCPUDescriptorHandleForHeapStart();
        auto handleGPU = pStream->pDescHeap->GetGPUDescriptorHandleForHeapStart();
        auto nStep = device->pDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        handleCPU.ptr += pStream->descIdxOffset * nStep;
        handleGPU.ptr += pStream->descIdxOffset * nStep;
        pStream->descIdxOffset += (uint32_t)hd->inputs.size() + (uint32_t)hd->outputs.size();

        // Create SRV and UAVs at shader launch time.
        // A higher performance solution may be pre-create it in CPU desc heaps and then copy the desc to GPU heaps in realtime.
        for (size_t i = 0; i < hd->inputs.size(); ++i)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc;
            ZeroMemory(&uavDesc, sizeof(uavDesc));
            uavDesc.Format = DXGI_FORMAT_UNKNOWN;
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            assert(offsets[i] % (uint32_t)hd->inputs[i].TypeSize() == 0);
            uavDesc.Buffer.FirstElement = offsets[i] / (uint32_t)hd->inputs[i].TypeSize();
            if (_USE_DESCRIPTOR_HEAP_ != 2)
                uavDesc.Buffer.NumElements = hd->inputs[i].NumElements();
            else
                uavDesc.Buffer.NumElements = ((dx_buffer_t*)devicePtrs[i])->size / (uint32_t)hd->inputs[i].TypeSize() - uavDesc.Buffer.FirstElement;
            uavDesc.Buffer.StructureByteStride = (uint32_t)hd->inputs[i].TypeSize();
            device->pDevice->CreateUnorderedAccessView(((dx_buffer_t*)devicePtrs[i])->handle.Get(), nullptr, &uavDesc, handleCPU);
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
            if (_USE_DESCRIPTOR_HEAP_ != 2)
                uavDesc.Buffer.NumElements = (uint32_t)hd->outputs[i].NumElements();
            else
                uavDesc.Buffer.NumElements = ((dx_buffer_t*)devicePtrs[hd->inputs.size() + i])->size / (uint32_t)hd->outputs[i].TypeSize() - uavDesc.Buffer.FirstElement;
            uavDesc.Buffer.StructureByteStride = (uint32_t)hd->outputs[i].TypeSize();
            device->pDevice->CreateUnorderedAccessView(((dx_buffer_t*)devicePtrs[hd->inputs.size() + i])->handle.Get(), nullptr, &uavDesc, handleCPU);
            handleCPU.ptr += nStep;
        }
        pStream->pCmdList->SetComputeRootDescriptorTable(0, handleGPU);
        if (hd->cbuffer_sizes.size() > 1)
            pStream->pCmdList->SetComputeRoot32BitConstants(1, pargs.size(), pargs.data(), 0);
    }
    else
    {

        for (uint32_t i = 0; i < hd->inputs.size(); ++i)
            pStream->pCmdList->SetComputeRootUnorderedAccessView(i, ((dx_buffer_t*)devicePtrs[i])->handle.Get()->GetGPUVirtualAddress() + offsets[i]);
        for (uint32_t i = 0; i < hd->outputs.size(); ++i)
            pStream->pCmdList->SetComputeRootUnorderedAccessView((UINT)hd->inputs.size() + i, ((dx_buffer_t*)devicePtrs[hd->inputs.size() + i])->handle.Get()->GetGPUVirtualAddress() + offsets[hd->inputs.size() + i]);
        if (hd->cbuffer_sizes.size() > 1)
            pStream->pCmdList->SetComputeRoot32BitConstants(hd->inputs.size() + hd->outputs.size(), pargs.size(), pargs.data(), 0);
    }

#ifdef _USE_GPU_TIMER_
    int m_nTimerIndex = device->AllocTimerIndex();
    // Set StartTimer here to only consider kernel execution time.
    device->StartTimer(pStream->pCmdList.Get(), m_nTimerIndex);
#endif
    pStream->pCmdList->Dispatch(blocks >= 0 ? blocks : hd->block[0], hd->block[1], hd->block[2]);
#ifdef _USE_GPU_TIMER_
    device->StopTimer(pStream->pCmdList.Get(), m_nTimerIndex);
#endif
    return 0;
}

int dxShaderLaunchAsync(void* hShader, void** buffers, void* hStream)
{
    return dxShaderLaunchAsyncExt(hShader, buffers, -1, hStream);
}

void* dxEventCreate()
{
    DEBUG_PRINT(__func__);

    if (dxInit() != 0)
        return nullptr;

    // Return available query slots.
    if (device->globalFreeQueries.size() > 0)
    {
        auto ret = device->globalFreeQueries.back().release();
        device->globalFreeQueries.pop_back();
        return ret;
    }

    // If no free heaps, create new heap
    if (device->globalQueryHeaps.size() == 0 ||
        device->globalQueryHeaps.back().curIdx >= device->globalQueryHeaps.back().totSize)
    {
        antares::dx_query_heap_t qheap;
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

        IFE(device->pDevice->CreateCommittedResource(&HeapProps, D3D12_HEAP_FLAG_NONE, &BufferDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_GRAPHICS_PPV_ARGS(qheap.pReadbackBuffer.ReleaseAndGetAddressOf())));

        D3D12_QUERY_HEAP_DESC QueryHeapDesc;
        QueryHeapDesc.Count = MAX_QUERY_NUM;
        QueryHeapDesc.NodeMask = 1;
        QueryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        IFE(device->pDevice->CreateQueryHeap(&QueryHeapDesc, IID_GRAPHICS_PPV_ARGS(qheap.pHeap.ReleaseAndGetAddressOf())));

        qheap.curIdx = 0;
        qheap.totSize = MAX_QUERY_NUM;
        device->globalQueryHeaps.push_back(qheap);
    }

    // Assume heap has free slots. 
    auto ret = new antares::dx_query_t;
    ret->heapIdx = (uint32_t)device->globalQueryHeaps.size() - 1;
    ret->queryIdxInHeap = device->globalQueryHeaps.back().curIdx;
    device->globalQueryHeaps.back().curIdx++;
    return ret;
}

int dxEventDestroy(void* hEvent)
{
    DEBUG_PRINT(__func__);

    if (hEvent == nullptr)
        return -1;

    // We just push queries for reuse.
    // Since queries only consume little memory, we only actually release them when app exits.
    std::unique_ptr<antares::dx_query_t> q((antares::dx_query_t*)hEvent);
    device->globalFreeQueries.push_back(std::move(q));
    return 0;
}

int dxEventRecord(void* hEvent, void* hStream)
{
    DEBUG_PRINT(__func__);

    if (!hStream)
        hStream = defaultStream;

    auto pQuery = (antares::dx_query_t*)hEvent;
    auto pStream = (dx_stream_t*)hStream;
    // Record commandlist.
    pStream->pCmdList->EndQuery(
        device->globalQueryHeaps[pQuery->heapIdx].pHeap.Get(),
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

float dxEventElapsedSecond(void* hStart, void* hStop)
{
    DEBUG_PRINT(__func__);

    auto pQueryStart = (antares::dx_query_t*)hStart;
    auto pQueryEnd = (antares::dx_query_t*)hStop;

    // Map readback buffer and read out data, assume the query heaps have already been resolved.
    uint64_t* pData;
    uint64_t timeStampStart = 0;
    uint64_t timeStampEnd = 0;

    HRESULT res = device->globalQueryHeaps[pQueryStart->heapIdx].pReadbackBuffer->Map(0, nullptr, reinterpret_cast<void**>(&pData));
    if (res < 0)
        return -1.0f;

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
        IFE(device->globalQueryHeaps[pQueryEnd->heapIdx].pReadbackBuffer->Map(0, nullptr, reinterpret_cast<void**>(&pDataEnd)));
        timeStampEnd = pDataEnd[pQueryEnd->queryIdxInHeap];
        device->globalQueryHeaps[pQueryEnd->heapIdx].pReadbackBuffer->Unmap(0, nullptr);
    }
    device->globalQueryHeaps[pQueryStart->heapIdx].pReadbackBuffer->Unmap(0, nullptr);

    uint64_t GpuFrequency;
    IFE(device->pCommandQueue->GetTimestampFrequency(&GpuFrequency));
    return static_cast<float>(timeStampEnd - timeStampStart) / static_cast<float>(GpuFrequency);
}