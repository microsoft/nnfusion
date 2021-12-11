// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef __ANTARES_D3D12_WRAPPER__
#define __ANTARES_D3D12_WRAPPER__

#define __EXPORT__ extern "C" __declspec(dllexport)

__EXPORT__ int    dxInit(int flags);
__EXPORT__ int    dxFinalize();

__EXPORT__ void*  dxStreamCreate();
__EXPORT__ int    dxStreamDestroy(void* hStream);
__EXPORT__ int    dxStreamSubmit(void* hStream);
__EXPORT__ int    dxStreamSynchronize(void* hStream);

__EXPORT__ void*  dxMemAlloc(size_t bytes);
__EXPORT__ int    dxMemFree(void* dptr);
__EXPORT__ int    dxMemcpyHtoDAsync(void* dst, void* src, size_t bytes, void* hStream);
__EXPORT__ int    dxMemcpyDtoHAsync(void* dst, void* src, size_t bytes, void* hStream);
__EXPORT__ int    dxMemcpyDtoDAsync(void* dst, void* src, size_t bytes, void* hStream);

__EXPORT__ void*  dxModuleLoad(const char* module_src);
__EXPORT__ void*  dxModuleGetShader(void *hModule, const char* fname);
__EXPORT__ void   dxModuleUnload(void* hModule);

__EXPORT__ void*  dxShaderLoad_v2(const char* shader_src);
__EXPORT__ int    dxShaderLaunchAsync(void* hShader, void** buffers, void* hStream);
__EXPORT__ void   dxShaderUnload(void* hShader);

__EXPORT__ void*  dxEventCreate();
__EXPORT__ int    dxEventRecord(void* hEvent, void* hStream);
__EXPORT__ float  dxEventElapsedSecond(void* hStart, void* hStop);
__EXPORT__ int    dxEventDestroy(void* hEvent);

#endif