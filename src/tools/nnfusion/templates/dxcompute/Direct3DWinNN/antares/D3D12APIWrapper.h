// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef __ANTARES_D3D12_WRAPPER__
#define __ANTARES_D3D12_WRAPPER__

#ifdef ANTARES_EXPORTS
#define ANTARES_API __declspec(dllexport)
#else
#define ANTARES_API __declspec(dllimport)
#endif

extern "C" ANTARES_API	int		dxInit(int flags);
extern "C" ANTARES_API	int		dxFinalize();

extern "C" ANTARES_API	void*	dxStreamCreate();
extern "C" ANTARES_API	int		dxStreamDestroy(void* hStream);
extern "C" ANTARES_API	int		dxStreamSubmit(void* hStream);
extern "C" ANTARES_API	int		dxStreamSynchronize(void* hStream);

extern "C" ANTARES_API	void*	dxMemAlloc(size_t bytes);
extern "C" ANTARES_API	int		dxMemFree(void* dptr);
extern "C" ANTARES_API	int		dxMemcpyHtoDAsync(void* dst, void* src, size_t bytes, void* hStream);
extern "C" ANTARES_API	int		dxMemcpyDtoHAsync(void* dst, void* src, size_t bytes, void* hStream);
extern "C" ANTARES_API	int		dxMemcpyDtoDAsync(void* dst, void* src, size_t bytes, void* hStream);

extern "C" ANTARES_API	int		dxModuleSetCompat(const char* compat_name);
extern "C" ANTARES_API	void*	dxModuleLoad(const char* module_src);
extern "C" ANTARES_API	void*	dxModuleGetShader(void* hModule, const char* fname);
extern "C" ANTARES_API	void	dxModuleUnload(void* hModule);

extern "C" ANTARES_API	void*	dxShaderLoad_v2(const char* shader_src);
extern "C" ANTARES_API	int		dxShaderLaunchAsyncExt(void* hShader, void** buffers, int n, int blocks, void* hStream);
extern "C" ANTARES_API	int		dxShaderLaunchAsync(void* hShader, void** buffers, void* hStream);
extern "C" ANTARES_API	void	dxShaderUnload(void* hShader);

extern "C" ANTARES_API	void*	dxEventCreate();
extern "C" ANTARES_API	int		dxEventRecord(void* hEvent, void* hStream);
extern "C" ANTARES_API	float	dxEventElapsedSecond(void* hStart, void* hStop);
extern "C" ANTARES_API	int		dxEventDestroy(void* hEvent);


#endif