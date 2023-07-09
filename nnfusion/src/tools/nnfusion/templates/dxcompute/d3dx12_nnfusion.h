// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "d3dx12_helper.h"
#include <map>

#define ASSERT(x)  ((x) ? (printf("Error-line: (%s) %d\n", __FILE__, __LINE__), _exit(1), 0): 1)

namespace nnfusion_dml
{
    template <class T, class P>
    std::string read_file(P& printer, const T& name)
    {
        std::ifstream t(name, ios_base::binary);
        if (t.fail())
        {
            printer << "[Error] Cannot find file from: `" << name
                << "`, please copy the full codegen folder!" << std::endl;
            ASSERT(0);
        }
        std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        return std::move(str);
    }

    template <class T>
    std::vector<char> load_data(const std::string& name, size_t num_elements, const T defval = 1)
    {
        std::vector<char> ret(num_elements * sizeof(T));
        if (name == "")
        {
            auto hptr = (T*)ret.data();
            std::fill(hptr, hptr + num_elements, defval);
        }
        else
        {
            auto str = read_file(std::cout, "Constant\\" + name);
            assert(str.size() == num_elements * sizeof(T));
            memcpy(ret.data(), str.data(), str.size());
        }
        return std::move(ret);
    }

    static std::vector<ID3D12CommandList*> cmdQueue, preloadQueue;
    static std::map<std::wstring, ComPtr<ID3DBlob>> computeShaderDict;
    static std::map<std::wstring, std::pair<double, int>> profCostDict;
    static unsigned long totalGPUMemoryAccess = 0;

    class NNfusionTensor
    {
        ComPtr<ID3D12Resource> deviceGPUSrcX;
        std::vector<size_t> shape;
        size_t type_size;

    public:
        NNfusionTensor(D3DDevice& device, const std::vector<size_t>& shape, size_t type_size)
            : shape(shape)
            , type_size(type_size)
        {
            size_t size = type_size * NumElements();
            size = ((size - 1) | 1023) + 1;
            totalGPUMemoryAccess += size;
            device.CreateGPUOnlyResource(size, &deviceGPUSrcX);
        }

        size_t NumElements() const
        {
            return std::accumulate(shape.begin(), shape.end(), 1LU, std::multiplies<size_t>());
        }

        size_t TypeSize() const { return type_size; }
        ComPtr<ID3D12Resource> Data() const { return deviceGPUSrcX; }
        std::vector<size_t> Shape() const { return shape; }
    };


    class NNfusionMemcpy
    {
        ComPtr<ID3D12Resource> deviceGPUSrcX;
        ComPtr<ID3D12Resource> deviceCPUSrcX;
        ComPtr<ID3D12GraphicsCommandList> m_computeCommandList;
        size_t bufferSize, elements;

    public:
        NNfusionMemcpy(D3DDevice& device,
            NNfusionTensor& dst,
            const std::vector<char> &src, bool preload = false)
        {
            elements = dst.NumElements();
            bufferSize = dst.TypeSize() * dst.NumElements();
            bufferSize = ((bufferSize - 1) | 1023) + 1;

            deviceGPUSrcX = dst.Data();
            device.CreateUploadBuffer(bufferSize, &deviceCPUSrcX);
            device.MapAndCopyToResource(deviceCPUSrcX.Get(), src.data(), src.size());

            IFE(device.pDevice->CreateCommandList(0,
                D3D12_COMMAND_LIST_TYPE_COMPUTE,
                device.pCommandAllocator.Get(),
                nullptr,
                IID_PPV_ARGS(&m_computeCommandList)));
            m_computeCommandList->CopyResource(deviceGPUSrcX.Get(), deviceCPUSrcX.Get());
            m_computeCommandList->Close();

            if (preload)
            {
                preloadQueue.push_back(Launch());
                return;
            }
            cmdQueue.push_back(Launch());
        }

        NNfusionMemcpy(D3DDevice& device,
            void* dst,
            NNfusionTensor& src)
        {
            elements = src.NumElements();
            bufferSize = src.TypeSize() * src.NumElements();
            bufferSize = ((bufferSize - 1) | 1023) + 1;

            deviceGPUSrcX = src.Data();
            device.CreateReadbackBuffer(bufferSize, &deviceCPUSrcX);

            IFE(device.pDevice->CreateCommandList(0,
                D3D12_COMMAND_LIST_TYPE_COMPUTE,
                device.pCommandAllocator.Get(),
                nullptr,
                IID_PPV_ARGS(&m_computeCommandList)));
            m_computeCommandList->CopyResource(deviceCPUSrcX.Get(), deviceGPUSrcX.Get());
            m_computeCommandList->Close();
            cmdQueue.push_back(Launch());
        }

        ID3D12GraphicsCommandList* Launch() { return m_computeCommandList.Get(); }
        template <class T>
        void PrintStageBuffer(D3DDevice& device, const std::string& name)
        {
            assert(bufferSize % sizeof(T) == 0);
            std::vector<T> dst(bufferSize / sizeof(T));
            device.MapCopyFromResource(deviceCPUSrcX.Get(), dst.data(), bufferSize);
            T* buffer = (T*)dst.data();
            std::cout << "Result(" << name << ") = {";

            constexpr size_t most_display = 6L;
            for (int i = 0; i < min(elements, most_display); ++i)
            {
                if (i)
                    std::cout << ", ";
                std::cout << dst[i];
            }
            if (elements > most_display)
            {
                std::cout << " .., " << dst[elements - 1];
            }
            std::cout << "}\n" << std::endl;
        }
    };

    class NNfusionOperator
    {
        ComPtr<ID3D12GraphicsCommandList> m_computeCommandList;

        ComPtr<ID3D12RootSignature> m_computeRootSignature;
        ComPtr<ID3DBlob> computeShader;
        ComPtr<ID3D12PipelineState> m_computeState;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc;

        LPCWSTR hlsl_source;

    public:
        NNfusionOperator(D3DDevice& device,
            const std::vector<NNfusionTensor>& inputs,
            const std::vector<NNfusionTensor>& outputs,
            LPCWSTR hlsl_source)
            : hlsl_source(hlsl_source)
        {

#define _USE_DECRIPTOR_HEAP_

#ifdef _USE_DECRIPTOR_HEAP_

			struct DescHeap {
				ComPtr<ID3D12DescriptorHeap> heap;
				D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
				UINT nDescStep, offsetRecord;
			};

			static DescHeap globalDescHeap;
			static bool initHeap = false;
			if (!globalDescHeap.nDescStep) {
				initHeap = true;
				auto InitDescriptorHeap = [](ID3D12Device* pDevice, D3D12_DESCRIPTOR_HEAP_TYPE type, UINT nDescriptors)
				{
					D3D12_DESCRIPTOR_HEAP_DESC desc;
					memset(&desc, 0, sizeof(desc));
					ZeroMemory(&desc, sizeof(desc));
					desc.NumDescriptors = nDescriptors;
					desc.Type = type;
					desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
					ComPtr<ID3D12DescriptorHeap> pDescHeap;
					IFE(pDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&pDescHeap)));

					globalDescHeap.nDescStep = pDevice->GetDescriptorHandleIncrementSize(type);
					globalDescHeap.heap = pDescHeap;
					globalDescHeap.cpuHandle = pDescHeap->GetCPUDescriptorHandleForHeapStart();
					globalDescHeap.offsetRecord = 0;
				};

				const UINT MAX_HEAP_SIZE = (1U << 20);
				InitDescriptorHeap(device.pDevice.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, MAX_HEAP_SIZE);
				assert(globalDescHeap.nDescStep > 0);
			}

			std::vector<UINT> argsOffset;
			// Prepare Heap Argument Offset
			for (int i = 0; i < inputs.size(); ++i) {
				D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
				ZeroMemory(&srvDesc, sizeof(srvDesc));
				srvDesc.Format = DXGI_FORMAT_UNKNOWN;
				srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
				srvDesc.Buffer.FirstElement = 0;
				srvDesc.Buffer.NumElements = inputs[i].NumElements();
				srvDesc.Buffer.StructureByteStride = inputs[i].TypeSize();
				srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

				device.pDevice->CreateShaderResourceView(inputs[i].Data().Get(), &srvDesc, globalDescHeap.cpuHandle);
				globalDescHeap.cpuHandle.ptr += globalDescHeap.nDescStep;
				argsOffset.push_back(globalDescHeap.offsetRecord++);
				assert(globalDescHeap.offsetRecord <= MAX_HEAP_SIZE);
			}
			for (int i = 0; i < outputs.size(); ++i) {
				D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc;
				ZeroMemory(&uavDesc, sizeof(uavDesc));
				uavDesc.Format = DXGI_FORMAT_UNKNOWN;
				uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
				uavDesc.Buffer.FirstElement = 0;
				uavDesc.Buffer.NumElements = outputs[i].NumElements();
				uavDesc.Buffer.StructureByteStride = outputs[i].TypeSize();
				device.pDevice->CreateUnorderedAccessView(outputs[i].Data().Get(), nullptr, &uavDesc, globalDescHeap.cpuHandle);
				globalDescHeap.cpuHandle.ptr += globalDescHeap.nDescStep;
				argsOffset.push_back(globalDescHeap.offsetRecord++);
				assert(globalDescHeap.offsetRecord <= MAX_HEAP_SIZE);
			}

			// Prepare Root
			std::vector<CD3DX12_ROOT_PARAMETER1> computeRootParameters(1);
			CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
			// D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE is needed to disable unproper driver optimization.
			ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, inputs.size(), 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, argsOffset[0]);
			ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, outputs.size(), 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, argsOffset[inputs.size()]);

			computeRootParameters[0].InitAsDescriptorTable(2, ranges);
			CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
			computeRootSignatureDesc.Init_1_1((UINT)computeRootParameters.size(),
				computeRootParameters.data());
#else
            // Prepare Root
            std::vector<CD3DX12_ROOT_PARAMETER1> computeRootParameters(inputs.size() +
                outputs.size());
            for (int i = 0; i < inputs.size(); ++i)
                computeRootParameters[i].InitAsShaderResourceView(i);
            for (int i = 0; i < outputs.size(); ++i)
                computeRootParameters[inputs.size() + i].InitAsUnorderedAccessView(i);

            CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
            computeRootSignatureDesc.Init_1_1(computeRootParameters.size(),
                computeRootParameters.data());
#endif

            ComPtr<ID3DBlob> signature;
            ComPtr<ID3DBlob> error;

            IFE(D3DX12SerializeVersionedRootSignature(
                &computeRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error));
            IFE(device.pDevice->CreateRootSignature(0,
                signature->GetBufferPointer(),
                signature->GetBufferSize(),
                IID_PPV_ARGS(&m_computeRootSignature)));

            auto path = std::wstring(L"HLSL\\") + hlsl_source;
            std::wcout << L"[Info] Loading HLSL data from: `" << path << L"` .." << std::endl;
            auto str = read_file(std::wcout, path);
            int at_bx = str.find("// [thread_extent] blockIdx.x = "), blockX = (at_bx >= 0) ? std::atoi(str.data() + at_bx + sizeof("// [thread_extent] blockIdx.x = ") - 1) : 1;
            int at_by = str.find("// [thread_extent] blockIdx.y = "), blockY = (at_by >= 0) ? std::atoi(str.data() + at_by + sizeof("// [thread_extent] blockIdx.y = ") - 1) : 1;
            int at_bz = str.find("// [thread_extent] blockIdx.z = "), blockZ = (at_bz >= 0) ? std::atoi(str.data() + at_bz + sizeof("// [thread_extent] blockIdx.z = ") - 1) : 1;
            std::vector<UINT> threads = { (UINT)blockX, (UINT)blockY, (UINT)blockZ };

            auto it = computeShaderDict.find(hlsl_source);
            if (it == computeShaderDict.end())
            {
                IFE(D3DCompileFromFile(
                    path.c_str(), NULL, NULL, "CSMain", "cs_5_0", 0, 0, &computeShader, NULL));
                computeShaderDict[hlsl_source] = computeShader;
            }
            else
                computeShader = it->second;

            computePsoDesc.pRootSignature = m_computeRootSignature.Get();
            computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());

            IFE(device.pDevice->CreateComputePipelineState(&computePsoDesc,
                IID_PPV_ARGS(&m_computeState)));
            IFE(device.pDevice->CreateCommandList(0,
                D3D12_COMMAND_LIST_TYPE_COMPUTE,
                device.pCommandAllocator.Get(),
                m_computeState.Get(),
                IID_PPV_ARGS(&m_computeCommandList)));

            m_computeCommandList->SetComputeRootSignature(m_computeRootSignature.Get());

#ifdef _USE_DECRIPTOR_HEAP_
			ID3D12DescriptorHeap* pHeaps[] = { globalDescHeap.heap.Get() };
			m_computeCommandList->SetDescriptorHeaps(1, pHeaps);
			m_computeCommandList->SetComputeRootDescriptorTable(0, globalDescHeap.heap->GetGPUDescriptorHandleForHeapStart());
#else
            for (int i = 0; i < inputs.size(); ++i)
                m_computeCommandList->SetComputeRootShaderResourceView(
                    i, inputs[i].Data()->GetGPUVirtualAddress());
            for (int i = 0; i < outputs.size(); ++i)
                m_computeCommandList->SetComputeRootUnorderedAccessView(
                    inputs.size() + i, outputs[i].Data()->GetGPUVirtualAddress());
#endif
            m_computeCommandList->Dispatch(threads[0], threads[1], threads[2]);
            IFE(m_computeCommandList->Close());

            cmdQueue.push_back(Launch());

            if (!profCostDict.count(hlsl_source)) {
                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                constexpr int NUM_STEPS = 10;
                for (int i = 0; i < NUM_STEPS; i++)
                {
                    device.pCommandQueue->ExecuteCommandLists(1, cmdQueue.data() + cmdQueue.size() - 1);
                    device.AwaitExecution();
                }
                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() /
                    NUM_STEPS;
                profCostDict[hlsl_source] = { sec, 1 };
            }
            else
                profCostDict[hlsl_source].second++;
        }

        ID3D12GraphicsCommandList* Launch() { return m_computeCommandList.Get(); }
    };
}

