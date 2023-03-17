// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef __ANTARES_D3D12_WRAPPER__
#define __ANTARES_D3D12_WRAPPER__

#ifdef ANTARES_EXPORTS
#define ANTARES_API __declspec(dllexport)
#else
#define ANTARES_API __declspec(dllimport)
#endif

#define __EXPORT__ extern "C"

__EXPORT__ ANTARES_API int    dxInit(int flags, int ord);
__EXPORT__ ANTARES_API int    dxFinalize();

__EXPORT__ ANTARES_API void*  dxStreamCreate();
__EXPORT__ ANTARES_API int    dxStreamDestroy(void* hStream);
__EXPORT__ ANTARES_API int    dxStreamSubmit(void* hStream);
__EXPORT__ ANTARES_API int    dxStreamSynchronize(void* hStream);

__EXPORT__ ANTARES_API void*  dxMemAlloc(size_t bytes);
__EXPORT__ ANTARES_API int    dxMemFree(void* dptr);
__EXPORT__ ANTARES_API int    dxMemcpyHtoDAsync(void* dst, void* src, size_t bytes, void* hStream);
__EXPORT__ ANTARES_API int    dxMemcpyDtoHAsync(void* dst, void* src, size_t bytes, void* hStream);
__EXPORT__ ANTARES_API int    dxMemcpyDtoDAsync(void* dst, void* src, size_t bytes, void* hStream);

__EXPORT__ ANTARES_API void*  dxMemHostRegister(void* dptr, unsigned int subres);
__EXPORT__ ANTARES_API void   dxMemHostUnregister(void* dptr, unsigned int subres);

__EXPORT__ ANTARES_API const char* dxModuleCompile(const char* module_src, size_t* out_size);

__EXPORT__ ANTARES_API int    dxModuleSetCompat(const char* compat_name);
__EXPORT__ ANTARES_API void*  dxModuleLoad(const char* module_src);
__EXPORT__ ANTARES_API void*  dxModuleGetShader(void *hModule, const char* fname);
__EXPORT__ ANTARES_API void   dxModuleUnload(void* hModule);

__EXPORT__ ANTARES_API void*  dxShaderLoad_v3(const char* shader_src);
__EXPORT__ ANTARES_API int    dxShaderLaunchAsyncExt(void* hShader, void** buffers, int blocks, void* hStream);
__EXPORT__ ANTARES_API int    dxShaderLaunchAsync(void* hShader, void** buffers, void* hStream);
__EXPORT__ ANTARES_API void   dxShaderUnload(void* hShader);

__EXPORT__ ANTARES_API void*  dxEventCreate();
__EXPORT__ ANTARES_API int    dxEventRecord(void* hEvent, void* hStream);
__EXPORT__ ANTARES_API float  dxEventElapsedSecond(void* hStart, void* hStop);
__EXPORT__ ANTARES_API int    dxEventDestroy(void* hEvent);

#endif