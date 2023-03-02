// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <stdio.h>
#include <stdint.h>

#include <cassert>
#include <vector>
#include <wrl/client.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <direct.h>

#ifdef _GAMING_XBOX_SCARLETT
#include "pch.h"
#define _USE_DXC_
#include <dxcapi_xs.h>
#pragma comment(lib, "dxcompiler_xs.lib")

using namespace DirectX;
using namespace DX;

#else
#include <dxgi1_5.h>
#include <d3d12.h>
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

#ifdef _USE_DXC_
#include <dxcapi.h>
#pragma comment(lib, "dxcompiler.lib")
#else
#include <d3dcompiler.h>
#pragma comment(lib, "d3dcompiler.lib")
#endif
#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS

#endif

using namespace std;
using namespace Microsoft::WRL;


string error_string(HRESULT x, string FILE, int LINE)
{
    static string e_str;
    e_str = "Error-line: (" + FILE + ") " + std::to_string(LINE) + "\n\n";
    if (x == D3D12_ERROR_ADAPTER_NOT_FOUND) e_str += "Reason: The specified cached PSO was created on a different adapter and cannot be reused on the current adapter.\n";
    else if (x == D3D12_ERROR_DRIVER_VERSION_MISMATCH) e_str += "Reason: The specified cached PSO was created on a different driver version and cannot be reused on the current adapter.\n";
    else if (x == DXGI_ERROR_INVALID_CALL) e_str += "Reason: The method call is invalid. For example, a method's parameter may not be a valid pointer.\n";
    else if (x == DXGI_ERROR_WAS_STILL_DRAWING) e_str += "Reason: The previous blit operation that is transferring information to or from this surface is incomplete.\n";
    else if (x == E_FAIL) e_str += "Reason: Attempted to create a device with the debug layer enabled and the layer is not installed.\n";
    else if (x == E_INVALIDARG) e_str += "Reason: An invalid parameter was passed to the returning function.\n";
    else if (x == E_OUTOFMEMORY) e_str += "Reason: Direct3D could not allocate sufficient memory to complete the call.\n";
    else if (x == E_NOTIMPL) e_str += "Reason: The method call isn't implemented with the passed parameter combination.\n";
    else if (x == S_FALSE) e_str += "Reason: Alternate success value, indicating a successful but nonstandard completion (the precise meaning depends on context).\n";
    else
    {
        std::stringstream stream;
        stream << "0x"
            << std::setfill('0') << std::setw(sizeof(x) * 2)
            << std::hex << x;
        e_str += "Unknown reason, d3d error code: " + stream.str() + ".\n";
    }
    return e_str;
}

#define IFE(x)  ((FAILED(x)) ? (fprintf(stderr, error_string(x, __FILE__, __LINE__).c_str()), abort(), 0): 1)

namespace {

    inline const D3D12_COMMAND_QUEUE_DESC D3D12CommandQueueDesc(D3D12_COMMAND_LIST_TYPE type, D3D12_COMMAND_QUEUE_FLAGS flags = D3D12_COMMAND_QUEUE_FLAG_NONE, UINT nodeMask = 0, INT priority = 0)
    {
        D3D12_COMMAND_QUEUE_DESC desc = {
            type,
            priority,
            flags,
            nodeMask
        };
        return desc;
    }

    inline const D3D12_HEAP_PROPERTIES D3D12HeapProperties(
        D3D12_HEAP_TYPE heapType,
        D3D12_CPU_PAGE_PROPERTY pageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL memoryPoolType = D3D12_MEMORY_POOL_UNKNOWN,
        UINT creationNodeMask = 0,
        UINT visibleNodeMask = 0
    )
    {
        D3D12_HEAP_PROPERTIES heapProperties = {
            heapType,
            pageProperty,
            memoryPoolType,
            creationNodeMask,
            visibleNodeMask
        };
        return heapProperties;
    }

    inline const D3D12_RESOURCE_DESC D3D12BufferResourceDesc(
        UINT64 width,
        UINT height = 1,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        UINT64 alignment = 0
    )
    {
        D3D12_RESOURCE_DESC resourceDesc = {
            D3D12_RESOURCE_DIMENSION_BUFFER,
            alignment,
            width,
            height,
            1,
            1,
            DXGI_FORMAT_UNKNOWN,
            {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            flags
        };

        return resourceDesc;
    }

#ifndef _GAMING_XBOX_SCARLETT

    struct CD3DX12_ROOT_DESCRIPTOR_TABLE1 : public D3D12_ROOT_DESCRIPTOR_TABLE1
    {
        CD3DX12_ROOT_DESCRIPTOR_TABLE1() = default;
        explicit CD3DX12_ROOT_DESCRIPTOR_TABLE1(const D3D12_ROOT_DESCRIPTOR_TABLE1& o) :
            D3D12_ROOT_DESCRIPTOR_TABLE1(o)
        {}
        CD3DX12_ROOT_DESCRIPTOR_TABLE1(
            UINT numDescriptorRanges,
            _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* _pDescriptorRanges)
        {
            Init(numDescriptorRanges, _pDescriptorRanges);
        }

        inline void Init(
            UINT numDescriptorRanges,
            _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* _pDescriptorRanges)
        {
            Init(*this, numDescriptorRanges, _pDescriptorRanges);
        }

        static inline void Init(
            _Out_ D3D12_ROOT_DESCRIPTOR_TABLE1& rootDescriptorTable,
            UINT numDescriptorRanges,
            _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* _pDescriptorRanges)
        {
            rootDescriptorTable.NumDescriptorRanges = numDescriptorRanges;
            rootDescriptorTable.pDescriptorRanges = _pDescriptorRanges;
        }
    };

    struct CD3DX12_ROOT_CONSTANTS : public D3D12_ROOT_CONSTANTS
    {
        CD3DX12_ROOT_CONSTANTS() = default;
        explicit CD3DX12_ROOT_CONSTANTS(const D3D12_ROOT_CONSTANTS& o) :
            D3D12_ROOT_CONSTANTS(o)
        {}
        CD3DX12_ROOT_CONSTANTS(
            UINT num32BitValues,
            UINT shaderRegister,
            UINT registerSpace = 0)
        {
            Init(num32BitValues, shaderRegister, registerSpace);
        }

        inline void Init(
            UINT num32BitValues,
            UINT shaderRegister,
            UINT registerSpace = 0)
        {
            Init(*this, num32BitValues, shaderRegister, registerSpace);
        }

        static inline void Init(
            _Out_ D3D12_ROOT_CONSTANTS& rootConstants,
            UINT num32BitValues,
            UINT shaderRegister,
            UINT registerSpace = 0)
        {
            rootConstants.Num32BitValues = num32BitValues;
            rootConstants.ShaderRegister = shaderRegister;
            rootConstants.RegisterSpace = registerSpace;
        }
    };

    struct CD3DX12_ROOT_DESCRIPTOR1 : public D3D12_ROOT_DESCRIPTOR1
    {
        CD3DX12_ROOT_DESCRIPTOR1() = default;
        explicit CD3DX12_ROOT_DESCRIPTOR1(const D3D12_ROOT_DESCRIPTOR1& o) :
            D3D12_ROOT_DESCRIPTOR1(o)
        {}
        CD3DX12_ROOT_DESCRIPTOR1(
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE)
        {
            Init(shaderRegister, registerSpace, flags);
        }

        inline void Init(
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE)
        {
            Init(*this, shaderRegister, registerSpace, flags);
        }

        static inline void Init(
            _Out_ D3D12_ROOT_DESCRIPTOR1& table,
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE)
        {
            table.ShaderRegister = shaderRegister;
            table.RegisterSpace = registerSpace;
            table.Flags = flags;
        }
    };

    struct CD3DX12_ROOT_PARAMETER1 : public D3D12_ROOT_PARAMETER1
    {
        CD3DX12_ROOT_PARAMETER1() = default;
        explicit CD3DX12_ROOT_PARAMETER1(const D3D12_ROOT_PARAMETER1& o) :
            D3D12_ROOT_PARAMETER1(o)
        {}

        static inline void InitAsDescriptorTable(
            _Out_ D3D12_ROOT_PARAMETER1& rootParam,
            UINT numDescriptorRanges,
            _In_reads_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* pDescriptorRanges,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParam.ShaderVisibility = visibility;
            CD3DX12_ROOT_DESCRIPTOR_TABLE1::Init(rootParam.DescriptorTable, numDescriptorRanges, pDescriptorRanges);
        }

        static inline void InitAsConstants(
            _Out_ D3D12_ROOT_PARAMETER1& rootParam,
            UINT num32BitValues,
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            rootParam.ShaderVisibility = visibility;
            CD3DX12_ROOT_CONSTANTS::Init(rootParam.Constants, num32BitValues, shaderRegister, registerSpace);
        }

        static inline void InitAsConstantBufferView(
            _Out_ D3D12_ROOT_PARAMETER1& rootParam,
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            rootParam.ShaderVisibility = visibility;
            CD3DX12_ROOT_DESCRIPTOR1::Init(rootParam.Descriptor, shaderRegister, registerSpace, flags);
        }

        static inline void InitAsShaderResourceView(
            _Out_ D3D12_ROOT_PARAMETER1& rootParam,
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParam.ShaderVisibility = visibility;
            CD3DX12_ROOT_DESCRIPTOR1::Init(rootParam.Descriptor, shaderRegister, registerSpace, flags);
        }

        static inline void InitAsUnorderedAccessView(
            _Out_ D3D12_ROOT_PARAMETER1& rootParam,
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParam.ShaderVisibility = visibility;
            CD3DX12_ROOT_DESCRIPTOR1::Init(rootParam.Descriptor, shaderRegister, registerSpace, flags);
        }

        inline void InitAsDescriptorTable(
            UINT numDescriptorRanges,
            _In_reads_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* pDescriptorRanges,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            InitAsDescriptorTable(*this, numDescriptorRanges, pDescriptorRanges, visibility);
        }

        inline void InitAsConstants(
            UINT num32BitValues,
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            InitAsConstants(*this, num32BitValues, shaderRegister, registerSpace, visibility);
        }

        inline void InitAsConstantBufferView(
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            InitAsConstantBufferView(*this, shaderRegister, registerSpace, flags, visibility);
        }

        inline void InitAsShaderResourceView(
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            InitAsShaderResourceView(*this, shaderRegister, registerSpace, flags, visibility);
        }

        inline void InitAsUnorderedAccessView(
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            InitAsUnorderedAccessView(*this, shaderRegister, registerSpace, flags, visibility);
        }
    };

    struct CD3DX12_DESCRIPTOR_RANGE1 : public D3D12_DESCRIPTOR_RANGE1
    {
        CD3DX12_DESCRIPTOR_RANGE1() = default;
        explicit CD3DX12_DESCRIPTOR_RANGE1(const D3D12_DESCRIPTOR_RANGE1& o) :
            D3D12_DESCRIPTOR_RANGE1(o)
        {}
        CD3DX12_DESCRIPTOR_RANGE1(
            D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
            UINT numDescriptors,
            UINT baseShaderRegister,
            UINT registerSpace = 0,
            D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
            UINT offsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND)
        {
            Init(rangeType, numDescriptors, baseShaderRegister, registerSpace, flags, offsetInDescriptorsFromTableStart);
        }

        inline void Init(
            D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
            UINT numDescriptors,
            UINT baseShaderRegister,
            UINT registerSpace = 0,
            D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
            UINT offsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND)
        {
            Init(*this, rangeType, numDescriptors, baseShaderRegister, registerSpace, flags, offsetInDescriptorsFromTableStart);
        }

        static inline void Init(
            _Out_ D3D12_DESCRIPTOR_RANGE1& range,
            D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
            UINT numDescriptors,
            UINT baseShaderRegister,
            UINT registerSpace = 0,
            D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
            UINT offsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND)
        {
            range.RangeType = rangeType;
            range.NumDescriptors = numDescriptors;
            range.BaseShaderRegister = baseShaderRegister;
            range.RegisterSpace = registerSpace;
            range.Flags = flags;
            range.OffsetInDescriptorsFromTableStart = offsetInDescriptorsFromTableStart;
        }
    };

    struct CD3DX12_SHADER_BYTECODE : public D3D12_SHADER_BYTECODE
    {
        CD3DX12_SHADER_BYTECODE() = default;
        explicit CD3DX12_SHADER_BYTECODE(const D3D12_SHADER_BYTECODE& o) :
            D3D12_SHADER_BYTECODE(o)
        {}
        CD3DX12_SHADER_BYTECODE(
            _In_ ID3DBlob* pShaderBlob)
        {
            pShaderBytecode = pShaderBlob->GetBufferPointer();
            BytecodeLength = pShaderBlob->GetBufferSize();
        }
        CD3DX12_SHADER_BYTECODE(
            const void* _pShaderBytecode,
            SIZE_T bytecodeLength)
        {
            pShaderBytecode = _pShaderBytecode;
            BytecodeLength = bytecodeLength;
        }
    };

    struct CD3DX12_DEFAULT {};
    extern const DECLSPEC_SELECTANY CD3DX12_DEFAULT D3D12_DEFAULT;

    struct CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC : public D3D12_VERSIONED_ROOT_SIGNATURE_DESC
    {
        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC() = default;
        explicit CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& o) :
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC(o)
        {}
        explicit CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(const D3D12_ROOT_SIGNATURE_DESC& o)
        {
            Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
            Desc_1_0 = o;
        }
        explicit CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(const D3D12_ROOT_SIGNATURE_DESC1& o)
        {
            Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
            Desc_1_1 = o;
        }
        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            Init_1_0(numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
        }
        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER1* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            Init_1_1(numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
        }
        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(CD3DX12_DEFAULT)
        {
            Init_1_1(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
        }

        inline void Init_1_0(
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            Init_1_0(*this, numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
        }

        static inline void Init_1_0(
            _Out_ D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc,
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
            desc.Desc_1_0.NumParameters = numParameters;
            desc.Desc_1_0.pParameters = _pParameters;
            desc.Desc_1_0.NumStaticSamplers = numStaticSamplers;
            desc.Desc_1_0.pStaticSamplers = _pStaticSamplers;
            desc.Desc_1_0.Flags = flags;
        }

        inline void Init_1_1(
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER1* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            Init_1_1(*this, numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
        }

        static inline void Init_1_1(
            _Out_ D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc,
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER1* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
            desc.Desc_1_1.NumParameters = numParameters;
            desc.Desc_1_1.pParameters = _pParameters;
            desc.Desc_1_1.NumStaticSamplers = numStaticSamplers;
            desc.Desc_1_1.pStaticSamplers = _pStaticSamplers;
            desc.Desc_1_1.Flags = flags;
        }
    };

    struct CD3DX12_ROOT_SIGNATURE_DESC : public D3D12_ROOT_SIGNATURE_DESC
    {
        CD3DX12_ROOT_SIGNATURE_DESC() = default;
        explicit CD3DX12_ROOT_SIGNATURE_DESC(const D3D12_ROOT_SIGNATURE_DESC& o) :
            D3D12_ROOT_SIGNATURE_DESC(o)
        {}
        CD3DX12_ROOT_SIGNATURE_DESC(
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            Init(numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
        }
        CD3DX12_ROOT_SIGNATURE_DESC(CD3DX12_DEFAULT)
        {
            Init(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
        }

        inline void Init(
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            Init(*this, numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
        }

        static inline void Init(
            _Out_ D3D12_ROOT_SIGNATURE_DESC& desc,
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            desc.NumParameters = numParameters;
            desc.pParameters = _pParameters;
            desc.NumStaticSamplers = numStaticSamplers;
            desc.pStaticSamplers = _pStaticSamplers;
            desc.Flags = flags;
        }
    };

    inline HRESULT D3DX12SerializeVersionedRootSignature(
        _In_ const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* pRootSignatureDesc,
        D3D_ROOT_SIGNATURE_VERSION MaxVersion,
        _Outptr_ ID3DBlob** ppBlob,
        _Always_(_Outptr_opt_result_maybenull_) ID3DBlob** ppErrorBlob)
    {
        if (ppErrorBlob != nullptr)
        {
            *ppErrorBlob = nullptr;
        }

        switch (MaxVersion)
        {
        case D3D_ROOT_SIGNATURE_VERSION_1_0:
            switch (pRootSignatureDesc->Version)
            {
            case D3D_ROOT_SIGNATURE_VERSION_1_0:
                return D3D12SerializeRootSignature(&pRootSignatureDesc->Desc_1_0, D3D_ROOT_SIGNATURE_VERSION_1, ppBlob, ppErrorBlob);

            case D3D_ROOT_SIGNATURE_VERSION_1_1:
            {
                HRESULT hr = S_OK;
                const D3D12_ROOT_SIGNATURE_DESC1& desc_1_1 = pRootSignatureDesc->Desc_1_1;

                const SIZE_T ParametersSize = sizeof(D3D12_ROOT_PARAMETER) * desc_1_1.NumParameters;
                void* pParameters = (ParametersSize > 0) ? HeapAlloc(GetProcessHeap(), 0, ParametersSize) : nullptr;
                if (ParametersSize > 0 && pParameters == nullptr)
                {
                    hr = E_OUTOFMEMORY;
                }
                auto pParameters_1_0 = reinterpret_cast<D3D12_ROOT_PARAMETER*>(pParameters);

                if (SUCCEEDED(hr))
                {
                    for (UINT n = 0; n < desc_1_1.NumParameters; n++)
                    {
                        __analysis_assume(ParametersSize == sizeof(D3D12_ROOT_PARAMETER) * desc_1_1.NumParameters);
                        pParameters_1_0[n].ParameterType = desc_1_1.pParameters[n].ParameterType;
                        pParameters_1_0[n].ShaderVisibility = desc_1_1.pParameters[n].ShaderVisibility;

                        switch (desc_1_1.pParameters[n].ParameterType)
                        {
                        case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
                            pParameters_1_0[n].Constants.Num32BitValues = desc_1_1.pParameters[n].Constants.Num32BitValues;
                            pParameters_1_0[n].Constants.RegisterSpace = desc_1_1.pParameters[n].Constants.RegisterSpace;
                            pParameters_1_0[n].Constants.ShaderRegister = desc_1_1.pParameters[n].Constants.ShaderRegister;
                            break;

                        case D3D12_ROOT_PARAMETER_TYPE_CBV:
                        case D3D12_ROOT_PARAMETER_TYPE_SRV:
                        case D3D12_ROOT_PARAMETER_TYPE_UAV:
                            pParameters_1_0[n].Descriptor.RegisterSpace = desc_1_1.pParameters[n].Descriptor.RegisterSpace;
                            pParameters_1_0[n].Descriptor.ShaderRegister = desc_1_1.pParameters[n].Descriptor.ShaderRegister;
                            break;

                        case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE:
                            const D3D12_ROOT_DESCRIPTOR_TABLE1& table_1_1 = desc_1_1.pParameters[n].DescriptorTable;

                            const SIZE_T DescriptorRangesSize = sizeof(D3D12_DESCRIPTOR_RANGE) * table_1_1.NumDescriptorRanges;
                            void* pDescriptorRanges = (DescriptorRangesSize > 0 && SUCCEEDED(hr)) ? HeapAlloc(GetProcessHeap(), 0, DescriptorRangesSize) : nullptr;
                            if (DescriptorRangesSize > 0 && pDescriptorRanges == nullptr)
                            {
                                hr = E_OUTOFMEMORY;
                            }
                            auto pDescriptorRanges_1_0 = reinterpret_cast<D3D12_DESCRIPTOR_RANGE*>(pDescriptorRanges);

                            if (SUCCEEDED(hr))
                            {
                                for (UINT x = 0; x < table_1_1.NumDescriptorRanges; x++)
                                {
                                    __analysis_assume(DescriptorRangesSize == sizeof(D3D12_DESCRIPTOR_RANGE) * table_1_1.NumDescriptorRanges);
                                    pDescriptorRanges_1_0[x].BaseShaderRegister = table_1_1.pDescriptorRanges[x].BaseShaderRegister;
                                    pDescriptorRanges_1_0[x].NumDescriptors = table_1_1.pDescriptorRanges[x].NumDescriptors;
                                    pDescriptorRanges_1_0[x].OffsetInDescriptorsFromTableStart = table_1_1.pDescriptorRanges[x].OffsetInDescriptorsFromTableStart;
                                    pDescriptorRanges_1_0[x].RangeType = table_1_1.pDescriptorRanges[x].RangeType;
                                    pDescriptorRanges_1_0[x].RegisterSpace = table_1_1.pDescriptorRanges[x].RegisterSpace;
                                }
                            }

                            D3D12_ROOT_DESCRIPTOR_TABLE& table_1_0 = pParameters_1_0[n].DescriptorTable;
                            table_1_0.NumDescriptorRanges = table_1_1.NumDescriptorRanges;
                            table_1_0.pDescriptorRanges = pDescriptorRanges_1_0;
                        }
                    }
                }

                if (SUCCEEDED(hr))
                {
                    CD3DX12_ROOT_SIGNATURE_DESC desc_1_0(desc_1_1.NumParameters, pParameters_1_0, desc_1_1.NumStaticSamplers, desc_1_1.pStaticSamplers, desc_1_1.Flags);
                    hr = D3D12SerializeRootSignature(&desc_1_0, D3D_ROOT_SIGNATURE_VERSION_1, ppBlob, ppErrorBlob);
                }

                if (pParameters)
                {
                    for (UINT n = 0; n < desc_1_1.NumParameters; n++)
                    {
                        if (desc_1_1.pParameters[n].ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
                        {
                            HeapFree(GetProcessHeap(), 0, reinterpret_cast<void*>(const_cast<D3D12_DESCRIPTOR_RANGE*>(pParameters_1_0[n].DescriptorTable.pDescriptorRanges)));
                        }
                    }
                    HeapFree(GetProcessHeap(), 0, pParameters);
                }
                return hr;
            }
            }
            break;

        case D3D_ROOT_SIGNATURE_VERSION_1_1:
            return D3D12SerializeVersionedRootSignature(pRootSignatureDesc, ppBlob, ppErrorBlob);
        }

        return E_INVALIDARG;
    }
#endif
}

namespace antares {

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

    struct D3DDevice
    {
        ComPtr<ID3D12Device1> pDevice;
        ComPtr<ID3D12CommandQueue> pCommandQueue;
        ComPtr<ID3D12CommandAllocator> pCommandAllocator;
        ComPtr<ID3D12Fence> pFence;
        HANDLE event;
        uint64_t fenceValue = 0;
        bool bEnableDebugLayer = false;
        bool bEnableGPUValidation = false;

        // Allocate individual queries from heaps for higher efficiency.
        // Since they consume little memory, we can release heaps when app exits.
        std::vector<dx_query_heap_t> globalQueryHeaps;

        // Reuse queries since they are small objects and may be frequently created.
        // Use unique_ptr to grantee it will be released when app exits.
        std::vector<std::unique_ptr<dx_query_t>> globalFreeQueries;

        // GPU time stamp query doesn't work on some NVIDIA GPUs with specific drivers, so we switch to DIRECT queue.
#ifdef _USE_GPU_TIMER_
        static const D3D12_COMMAND_LIST_TYPE CommandListType = D3D12_COMMAND_LIST_TYPE_DIRECT;
#else
        static const D3D12_COMMAND_LIST_TYPE CommandListType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
#endif

        D3DDevice(bool EnableDebugLayer = false, bool EnableGPUValidation = false)
        {
            bEnableDebugLayer = EnableDebugLayer;
            bEnableGPUValidation = EnableGPUValidation;
        }

#ifndef _GAMING_XBOX_SCARLETT
        bool GetHardwareAdapter(int adapterIndex, IDXGIFactory4* pFactory, IDXGIAdapter1** ppAdapter)
        {
            *ppAdapter = nullptr;
            {
                IDXGIAdapter1* pAdapter = nullptr;
                HRESULT hr = pFactory->EnumAdapters1(adapterIndex, &pAdapter);
                if (hr == DXGI_ERROR_NOT_FOUND)
                  IFE(hr);

                if (SUCCEEDED(
                    D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice))))
                {
                    *ppAdapter = pAdapter;
                    return true;
                }
                pAdapter->Release();
            }
            return false;
        }
#endif

        void InitD3DDevice(int ord)
        {
#ifndef _GAMING_XBOX_SCARLETT
            ComPtr<IDXGIFactory4> factory;
            ComPtr<IDXGIAdapter1> hardwareAdapter;
            IFE(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
            if (!GetHardwareAdapter(ord, factory.Get(), &hardwareAdapter)) {
                IFE(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice)));
            }
#else
            // Create the DX12 API device object.
            D3D12XBOX_CREATE_DEVICE_PARAMETERS params = {};
            params.Version = D3D12_SDK_VERSION;
            if (bEnableDebugLayer) {
                // Enable the debug layer.
                params.ProcessDebugFlags = D3D12_PROCESS_DEBUG_FLAG_DEBUG_LAYER_ENABLED;
            }
            params.GraphicsCommandQueueRingSizeBytes = static_cast<UINT>(D3D12XBOX_DEFAULT_SIZE_BYTES);
            params.GraphicsScratchMemorySizeBytes = static_cast<UINT>(D3D12XBOX_DEFAULT_SIZE_BYTES);
            params.ComputeScratchMemorySizeBytes = static_cast<UINT>(D3D12XBOX_DEFAULT_SIZE_BYTES);

            HRESULT hr = D3D12XboxCreateDevice(
                nullptr,
                &params,
                IID_GRAPHICS_PPV_ARGS(pDevice.ReleaseAndGetAddressOf()));
            ThrowIfFailed(hr);
#endif
        }

        void Init(int ord)
        {
#ifndef _GAMING_XBOX_SCARLETT
            // Enable debug layer
            ComPtr<ID3D12Debug> pDebug;
            if (bEnableDebugLayer && SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pDebug))))
            {
                pDebug->EnableDebugLayer();

                ComPtr<ID3D12Debug1> pDebug1;
                if (bEnableGPUValidation && SUCCEEDED((pDebug->QueryInterface(IID_PPV_ARGS(&pDebug1)))))
                {
                    pDebug1->SetEnableGPUBasedValidation(true);
                }
            }
#endif

            InitD3DDevice(ord);

            // Create a command queue
            D3D12_COMMAND_QUEUE_DESC commandQueueDesc = D3D12CommandQueueDesc(CommandListType);
            IFE(pDevice->CreateCommandQueue(&commandQueueDesc, IID_GRAPHICS_PPV_ARGS(pCommandQueue.ReleaseAndGetAddressOf())));

            // Create a command allocator
            IFE(pDevice->CreateCommandAllocator(CommandListType, IID_GRAPHICS_PPV_ARGS(pCommandAllocator.ReleaseAndGetAddressOf())));

            // Create a CPU-GPU synchronization event
            event = CreateEvent(nullptr, FALSE, FALSE, nullptr);

            // Create a fence to allow GPU to signal upon completion of execution
            IFE(pDevice->CreateFence(fenceValue, D3D12_FENCE_FLAG_SHARED, IID_GRAPHICS_PPV_ARGS(pFence.ReleaseAndGetAddressOf())));

#ifdef _USE_GPU_TIMER_
#define _MAX_GPU_TIMER_ 65536
            InitProfilingResources(_MAX_GPU_TIMER_);
#endif
        }

        // Added fence related functions.
        uint64_t SignalFence()
        {
            // Signal
            ++fenceValue;
            IFE(pCommandQueue->Signal(pFence.Get(), fenceValue));
            return fenceValue;
        }
        void WaitForFence(uint64_t val)
        {
            if (pFence->GetCompletedValue() >= val)
                return;
            IFE(pFence->SetEventOnCompletion(val, event));
            DWORD retVal = WaitForSingleObject(event, INFINITE);
            if (retVal != WAIT_OBJECT_0)
            {
                DebugBreak();
            }
        }
        void AwaitExecution()
        {
            auto f = SignalFence();
            WaitForFence(f);
        }

        inline void CreateCommittedResource(
            const D3D12_HEAP_PROPERTIES& heapProperties,
            const D3D12_RESOURCE_DESC& resourceDesc,
            D3D12_RESOURCE_STATES initialState,
            ID3D12Resource** ppResource
        )
        {
            IFE(pDevice->CreateCommittedResource(
                &heapProperties,
                D3D12_HEAP_FLAG_NONE,
                &resourceDesc,
                initialState,
                nullptr,
                IID_GRAPHICS_PPV_ARGS(ppResource)
            ));
        }
        inline void CreateGPUOnlyResource(UINT64 size, ID3D12Resource** ppResource)
        {
            CreateCommittedResource(
                D3D12HeapProperties(D3D12_HEAP_TYPE_DEFAULT),
                D3D12BufferResourceDesc(size, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_COMMON,
                ppResource
            );
        }
        inline void CreateUploadBuffer(UINT64 size, ID3D12Resource** ppResource)
        {
            CreateCommittedResource(
                D3D12HeapProperties(D3D12_HEAP_TYPE_UPLOAD),
                D3D12BufferResourceDesc(size),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                ppResource
            );
        }

        inline void CreateReadbackBuffer(UINT64 size, ID3D12Resource** ppResource)
        {
            CreateCommittedResource(
                D3D12HeapProperties(D3D12_HEAP_TYPE_READBACK),
                D3D12BufferResourceDesc(size),
                D3D12_RESOURCE_STATE_COPY_DEST,
                ppResource
            );
        }

        inline void CreateDefaultBuffer(UINT64 size, ID3D12Resource** ppResource)
        {
            CreateCommittedResource(
                D3D12HeapProperties(D3D12_HEAP_TYPE_DEFAULT),
                D3D12BufferResourceDesc(size, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_COMMON,
                ppResource
            );
        }

        void MapAndCopyToResource(ID3D12Resource* pResource, const void* pSrc, UINT64 numBytes)
        {
            D3D12_RANGE range = { 0, static_cast<SIZE_T>(numBytes) };
            void* pData;
            IFE(pResource->Map(0, &range, reinterpret_cast<void**>(&pData)));
            memcpy(pData, pSrc, static_cast<SIZE_T>(numBytes));
            pResource->Unmap(0, &range);
        }

        void MapCopyFromResource(ID3D12Resource* pResource, void* pDest, UINT64 numBytes)
        {
            D3D12_RANGE range = { 0, static_cast<SIZE_T>(numBytes) };
            void* pData;
            IFE(pResource->Map(0, &range, reinterpret_cast<void**>(&pData)));
            memcpy(pDest, pData, static_cast<SIZE_T>(numBytes));
            pResource->Unmap(0, &range);
        }

#ifdef _USE_GPU_TIMER_
        // Profiling related resources
    public:
        uint32_t AllocTimerIndex() { return (m_nTimers++) % _MAX_GPU_TIMER_; }
        void StartTimer(ID3D12GraphicsCommandList* pCmdList, uint32_t nTimerIdx)
        {
            pCmdList->EndQuery(m_pQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, nTimerIdx * 2);
        }
        void StopTimer(ID3D12GraphicsCommandList* pCmdList, uint32_t nTimerIdx)
        {
            pCmdList->EndQuery(m_pQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, nTimerIdx * 2 + 1);
        }
        void SyncTimerData()
        {
            ID3D12CommandList* pCmdLists[] = { m_pResolveCmdList.Get() };
            pCommandQueue->ExecuteCommandLists(1, pCmdLists);
            AwaitExecution();
            uint64_t* pData;
            D3D12_RANGE range = { 0, m_nTimers * 2 * sizeof(uint64_t) };
            IFE(m_pReadBackBuffer->Map(0, &range, reinterpret_cast<void**>(&pData)));
            memcpy_s(m_TimerDataCPU.data(), sizeof(uint64_t) * m_nTimers * 2, pData, sizeof(uint64_t) * m_nTimers * 2);
            m_pReadBackBuffer->Unmap(0, nullptr);
        }

        double GetTime(uint32_t nTimerIdx)
        {
            assert(nTimerIdx < m_nTimers);
            uint64_t TimeStamp1 = m_TimerDataCPU[nTimerIdx * 2];
            uint64_t TimeStamp2 = m_TimerDataCPU[nTimerIdx * 2 + 1];
            return static_cast<double>(TimeStamp2 - TimeStamp1) * m_fGPUTickDelta;
        }

        std::vector<double> GetAllTimes()
        {
            std::vector<double> times;
            times.resize(m_nTimers);
            for (uint32_t i = 0; i < m_nTimers; ++i)
            {
                times[i] = GetTime(i);
            }
            return std::move(times);
        }

        // Lock GPU clock rate for more stable performance measurement.
        // Only works with Win10 developer mode.
        // Note that SetStablePowerState will disable GPU boost and potentially decrease GPU performance.
        // So don't use it in release version application.
        void LockGPUClock()
        {
            auto IsDeveloperModeEnabled = []()
            {
                HKEY hKey;
                if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock", 0, KEY_READ, &hKey) != ERROR_SUCCESS)
                {
                    return false;
                }
                DWORD value;
                DWORD nSize = sizeof(DWORD);
                if (RegQueryValueExW(hKey, L"AllowDevelopmentWithoutDevLicense", 0, NULL, reinterpret_cast<LPBYTE>(&value), &nSize) != ERROR_SUCCESS)
                {
                    RegCloseKey(hKey);
                    return false;
                }
                RegCloseKey(hKey);
                return value != 0;
            };
            if (IsDeveloperModeEnabled())
            {
                pDevice->SetStablePowerState(TRUE);
                printf("Win10 developer mode turned on, locked GPU clock.\n");
            }
        }

    private:
        void InitProfilingResources(uint32_t nMaxTimers)
        {
            uint64_t GpuFrequency;
            IFE(pCommandQueue->GetTimestampFrequency(&GpuFrequency));
            m_fGPUTickDelta = 1.0 / static_cast<double>(GpuFrequency);

            D3D12_HEAP_PROPERTIES HeapProps;
            HeapProps.Type = D3D12_HEAP_TYPE_READBACK;
            HeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            HeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            HeapProps.CreationNodeMask = 1;
            HeapProps.VisibleNodeMask = 1;

            D3D12_RESOURCE_DESC BufferDesc;
            BufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            BufferDesc.Alignment = 0;
            BufferDesc.Width = sizeof(uint64_t) * nMaxTimers * 2;
            BufferDesc.Height = 1;
            BufferDesc.DepthOrArraySize = 1;
            BufferDesc.MipLevels = 1;
            BufferDesc.Format = DXGI_FORMAT_UNKNOWN;
            BufferDesc.SampleDesc.Count = 1;
            BufferDesc.SampleDesc.Quality = 0;
            BufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            BufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

            IFE(pDevice->CreateCommittedResource(&HeapProps, D3D12_HEAP_FLAG_NONE, &BufferDesc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_GRAPHICS_PPV_ARGS(m_pReadBackBuffer.ReleaseAndGetAddressOf())));
            m_pReadBackBuffer->SetName(L"GpuTimeStamp Buffer");

            D3D12_QUERY_HEAP_DESC QueryHeapDesc;
            QueryHeapDesc.Count = nMaxTimers * 2;
            QueryHeapDesc.NodeMask = 1;
            QueryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
            IFE(pDevice->CreateQueryHeap(&QueryHeapDesc, IID_GRAPHICS_PPV_ARGS(m_pQueryHeap.ReleaseAndGetAddressOf())));
            m_pQueryHeap->SetName(L"GpuTimeStamp QueryHeap");

            IFE(pDevice->CreateCommandList(0,
                CommandListType,
                pCommandAllocator.Get(),
                nullptr,
                IID_GRAPHICS_PPV_ARGS(m_pResolveCmdList.ReleaseAndGetAddressOf())));
            m_pResolveCmdList->ResolveQueryData(m_pQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, nMaxTimers * 2, m_pReadBackBuffer.Get(), 0);
            m_pResolveCmdList->Close();
            m_nMaxTimers = nMaxTimers;
            m_nTimers = 0;
            m_TimerDataCPU.resize(nMaxTimers * 2);
        }

        Microsoft::WRL::ComPtr<ID3D12QueryHeap> m_pQueryHeap;
        Microsoft::WRL::ComPtr<ID3D12Resource> m_pReadBackBuffer;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_pResolveCmdList;
        double m_fGPUTickDelta = 0.0;
        uint32_t m_nMaxTimers = 0;
        uint32_t m_nTimers = 0;
        std::vector<uint64_t> m_TimerDataCPU;
#endif
    };

#ifdef _USE_DXC_
    class DXCompiler
    {
    public:
        static DXCompiler* Get()
        {
            static DXCompiler sm_compiler;
            return &sm_compiler;

        }

#ifdef _GAMING_XBOX_SCARLETT
        static std::vector<std::wstring> getXboxFlags(const std::string& text, std::string mode = "cu")
        {
            // Currently dxcompiler for xbox cann't work properly for some shaders thus hanging GPU.
            // Before compiler team's fix, two additional flags are added to explicitly prohibit generating these shaders.
            // 1. -D__XBOX_PER_THREAD_SCRATCH_SIZE_LIMIT_IN_BYTES=0, disable scratch usage
            // 2. -D__XBOX_MAX_VGPR_COUNT=[VGPR_COUNT], limit VGPR as below:
            //  ThreadGroup Size    Num Wave32s	Max VGPRs(CU_MODE)	Max VGPRs(wgp mode)
            //  256x1x1	    8	    256	                            256
            //  512x1x1	    16	    128	                            256
            //  1024x1x1	32	    64	                            128
            //  2048x1x1	64	    32	                            64

            std::vector<std::wstring> flags{ L"-D__XBOX_PER_THREAD_SCRATCH_SIZE_LIMIT_IN_BYTES=0" };

            // extract thread size from target snippet like [numthreads(870, 1, 1)]
            size_t threadSize = 1;
            size_t start = text.find("[numthreads(");
            if (start != string::npos)
            {
                size_t startX = start + strlen("[numthreads(");
                size_t endX = text.find(",", startX);
                size_t startY = endX + 1;
                size_t endY = text.find(",", startY);
                size_t startZ = endY + 1;
                size_t endZ = text.find(")", startZ);
                if (endX != string::npos && endY != string::npos && endZ != string::npos)
                {
                    size_t threadX = std::stoi(text.substr(startX, endX - startX));
                    size_t threadY = std::stoi(text.substr(startY, endY - startY));
                    size_t threadZ = std::stoi(text.substr(startZ, endZ - startZ));
                    threadSize = threadX * threadY * threadZ;
                }
            }

            static auto getVgprLimit = [](int threadSize, std::string mode) -> size_t
            {
                if (mode == "cu")
                {
                    if (threadSize <= 256)
                        return 256;
                    else if (threadSize <= 512)
                        return 128;
                    else if (threadSize <= 1024)
                        return 64;
                    else if (threadSize <= 2048)
                        return 32;
                    else
                        return 0;
                }
                else if (mode == "wgp")
                {
                    if (threadSize <= 256)
                        return 256;
                    else if (threadSize <= 512)
                        return 256;
                    else if (threadSize <= 1024)
                        return 128;
                    else if (threadSize <= 2048)
                        return 64;
                    else
                        return 0;
                }
                return 0;
            };

            size_t vgprLimit = getVgprLimit(threadSize, mode);
            if (vgprLimit > 0 && vgprLimit < 256)
            {
                flags.push_back(L"-D__XBOX_MAX_VGPR_COUNT=" + std::to_wstring(vgprLimit));
            }
            return flags;
        }
#endif

        ComPtr<IDxcBlob> Compile(LPCVOID pText, UINT32 size, LPCWSTR entryName, LPCWSTR profile)
        {
            ComPtr<IDxcBlob> pRet = nullptr;
            ComPtr<IDxcBlobEncoding> pSrcBlob;
            IFE(m_pLibrary->CreateBlobWithEncodingOnHeapCopy(pText, size, CP_UTF8, &pSrcBlob));
            ComPtr<IDxcOperationResult> pResult = nullptr;

            std::vector<const WCHAR*> args_i;
#ifdef _GAMING_XBOX_SCARLETT
            std::vector<std::wstring> args = getXboxFlags((char*)pText);
            for (size_t i = 0; i < args.size(); i++)
            {
                args_i.push_back(args[i].c_str());
            }
#endif
            if (std::wstring(profile) > std::wstring(L"cs_6_0"))
                args_i.push_back(L"-enable-16bit-types");
            if (std::wstring(profile) >= std::wstring(L"cs_6_5"))
                args_i.push_back(L"-enable-templates");
            args_i.push_back(L"-D__HLSL_SHADER_COMPUTE=1");
            // args_i.push_back(L"-O3");
            args_i.push_back(NULL);

            std::string errStr;
            if (SUCCEEDED(m_pCompiler->Compile(pSrcBlob.Get(), L"ShaderFile", entryName, profile, args_i.data(), args_i.size() - 1, NULL, 0, NULL, &pResult)))
            {
                HRESULT pStatus;
                if (SUCCEEDED(pResult->GetStatus(&pStatus)) && SUCCEEDED(pStatus))
                {
                    pResult->GetResult(&pRet);
                }
                else
                {
                    ComPtr<IDxcBlobEncoding> pErrorsBlob;
                    if (SUCCEEDED(pResult->GetErrorBuffer(&pErrorsBlob)) && pErrorsBlob)
                    {
                        errStr = std::string("Compilation Error:\n") + (char*)pErrorsBlob->GetBufferPointer() + "\n";
                    }
                }
            }
            else
            {
                if (pResult)
                {
                    ComPtr<IDxcBlobEncoding> pErrorsBlob;
                    if (SUCCEEDED(pResult->GetErrorBuffer(&pErrorsBlob)) && pErrorsBlob)
                    {
                        errStr = std::string("Compilation Error:\n") + (char*)pErrorsBlob->GetBufferPointer() + "\n";
                    }
                }
            }

            if (pRet == nullptr)
            {
                if (errStr.empty())
                {
                    errStr = "Compilation Error:\nUnknown error\n";
                }
                printf(errStr.c_str());
            }
            return pRet;
        }
    private:
        DXCompiler()
        {
            IFE(DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(m_pLibrary.ReleaseAndGetAddressOf())));
            IFE(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(m_pCompiler.ReleaseAndGetAddressOf())));
        }
        DXCompiler(const DXCompiler&) = delete;
        DXCompiler& operator=(const DXCompiler&) = delete;

        ComPtr<IDxcLibrary> m_pLibrary;
        ComPtr<IDxcCompiler> m_pCompiler;

    };
#endif
}
