// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef __ANTARES_D3D12_WRAPPER__
#define __ANTARES_D3D12_WRAPPER__

#define __EXPORT__ extern "C" __declspec(dllexport)

__EXPORT__ int    dxInit(int flags);

__EXPORT__ void*  dxStreamCreate();
__EXPORT__ int    dxStreamDestroy(void* hStream);
__EXPORT__ int    dxStreamSubmit(void* hStream);
__EXPORT__ int    dxStreamSynchronize(void* hStream);

__EXPORT__ void*  dxMemAlloc(size_t bytes);
__EXPORT__ int    dxMemFree(void* dptr);
__EXPORT__ int    dxMemcpyHtoDAsync(void* dst, void* src, size_t bytes, void* hStream);
__EXPORT__ int    dxMemcpyDtoHAsync(void* dst, void* src, size_t bytes, void* hStream);

__EXPORT__ void*  dxShaderLoad(const char* src, int* num_inputs, int* num_outputs);
__EXPORT__ int    dxShaderUnload(void* hShader);
__EXPORT__ int    dxShaderGetProperty(void* hShader, int arg_index, size_t* num_elements, size_t* type_size, const char** dtype_name);
__EXPORT__ int    dxShaderLaunchAsync(void* hShader, void** buffers, void* hStream);

__EXPORT__ void*  dxEventCreate();
__EXPORT__ int    dxEventRecord(void* hEvent, void* hStream);
__EXPORT__ float  dxEventElapsedTime(void* hStart, void* hStop);
__EXPORT__ int    dxEventDestroy(void* hEvent);

#endif