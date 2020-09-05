extern "C" __global__ void manual_dot_nn_op_float_m1_k512_n1024_kernel0(float* input0, float* input1, float* output0)
{
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int col_id = blockIdx.x * blockDim.x / 8 + lane_id;
    if (col_id < 1024)
    {
        float val = 0;
        int k_start = warp_id * 64;
        int k_end = (warp_id + 1) * 64;
        for (int i = k_start; i < k_end; i++)
        {
            val = fma(input0[i], input1[i * 1024 + col_id], val);
        }
        if (warp_id == 0)
        {
            output0[col_id]=0;
        }
        __syncthreads();
        atomicAdd(output0 + col_id, val);
    }

}
