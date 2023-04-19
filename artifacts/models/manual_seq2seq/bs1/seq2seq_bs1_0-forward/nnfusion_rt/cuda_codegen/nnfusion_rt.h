#pragma once

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
extern "C" int get_device_type();
extern "C" int kernel_entry(float* Parameter_13_0, int64_t* Parameter_14_0, float* Parameter_15_0, int64_t* Parameter_16_0, int64_t* Parameter_17_0, float* Parameter_18_0, float* Result_96_0, int64_t* Result_97_0, float* Result_98_0, int64_t* Result_99_0, int64_t* Result_100_0, char* Result_101_0);
extern "C" void cuda_init();
extern "C" void cuda_free();
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
