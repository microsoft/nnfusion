
// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the RUNTIME_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// RUNTIME_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef RUNTIME_EXPORTS
#define RUNTIME_API __declspec(dllexport)
#else
#define RUNTIME_API __declspec(dllimport)
#endif

extern "C" RUNTIME_API int get_device_type();
extern "C" RUNTIME_API int get_workspace_size();
extern "C" RUNTIME_API int kernel_entry(void* Parameter_0_0_host, void* Parameter_1_0_host, void* Result_3_0_host);
extern "C" RUNTIME_API void hlsl_init();
extern "C" RUNTIME_API void hlsl_free();
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

