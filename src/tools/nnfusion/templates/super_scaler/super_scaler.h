// Microsoft (c) 2019, NNFusion Team
#pragma once
/*
enum NNSCALER_DeviceType
{
    UNKNOWN,
    CPU,
    NVGPU,
    AMDGPU
};

struct NNSCALER_MemoryType
{
    DeviceType device_type, bool rdma,
};

struct NNSCALER_CommContext
{
    int globalRank;
    int globalSize;
    int localRank;
    int localSize;
    void* NCCLToken;
};
*/

void super_scaler_initialization();
void super_scaler_finalization();
void super_scaler_all_reduce(float* gradients,
                             float* out_gradients,
                             int size,
                             void* exestream = nullptr,
                             void (*callback)(void*) = nullptr,
                             void* callback_context = nullptr);
/*
void super_scaler_all_reduce_async(float* gradients,
                                   float* out_gradients,
                                   int size,
                                   void* exestream = nullptr,
                                   void (*callback)(void*) = nullptr,
                                   void* callback_context = nullptr);
*/

int super_scaler_get_localrank();
void super_scaler_sync();
/*
void nnscaler_all_reduce(float *gradient, float *gradient_out,  
    MemoryType memtype, size_t gradient_id, int size, void (*callback)(void*), void* cbctx, CommContext* commctx); 
    */
