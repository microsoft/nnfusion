// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
//c interface to init a superscaler session from plan file
void sc_init(const char* plan_path);

//c interface to destroy a superscaler session
void sc_finalize();

//c interface to check the number of participants in this session
void sc_get_world_size(int*);

//c interface to get current process uniq process id of all the participants in this session
void sc_get_host_id(int*);

//c interface to get current process uniq device id of all the participants in this session
void sc_get_device_id(int*);

// in-place allreduce, which means data's contents will change after allreduce
// tensor_name is used to index plans for this tensor
// if stream provided, superscaler will use it to do allreduce task
void sc_allreduce(const char* tensor_name, float* data, size_t size, void* stream);
void sc_send(const char* tensor_name, unsigned char* input, size_t size, void* stream);
void sc_recv(const char* tensor_name, unsigned char** output, size_t* size, void* stream);

#ifdef __cplusplus
}
#endif
