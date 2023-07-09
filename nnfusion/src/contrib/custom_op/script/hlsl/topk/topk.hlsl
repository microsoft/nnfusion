StructuredBuffer<__value_type__> input0: register(t0);
RWStructuredBuffer<__value_type__> output0: register(u0);
RWStructuredBuffer<__index_type__> output1: register(u1);
RWStructuredBuffer<int> output2: register(u2);

struct type {
    __value_type__ val;
    int index;
};

groupshared type buf[__thread_max_element__ * 2];

uint thread_id_to_idx(uint block_id, uint thread_id, uint axis_size, uint axis_stride)
{
    return (block_id / axis_stride) * (axis_size * axis_stride) + block_id % axis_stride + thread_id * axis_stride;
}

uint idx_to_thread_id(uint block_id, uint idx, uint axis_size, uint axis_stride)
{
    // return (block_id / axis_stride) * (axis_size * axis_stride) + block_id % axis_stride + thread_id * axis_stride;
    return (idx - (block_id / axis_stride) * (axis_size * axis_stride) - block_id % axis_stride) / axis_stride;
}

// Bitonic Sorting
void bitonic_sort(uint element_id, uint step, uint gstep, uint largest)
{
    uint pos = element_id % step + (element_id / step) * step * 2;
    if((((pos/gstep>>1 & 1) == largest) && buf[pos].val > buf[pos+step].val) || (((pos/gstep>>1 & 1) == (largest^1) && (buf[pos].val < buf[pos+step].val))))
    {
        type t =  buf[pos + step];
        buf[pos + step] = buf[pos];
        buf[pos] = t;
    }
}

void bitonic(uint element_id, uint step, uint gstep, uint largest)
{
    uint pos = element_id % step + (element_id / step) * step * 2;

}

[RootSignature("DescriptorTable(SRV(t0, numDescriptors=1), UAV(u0, numDescriptors=3))")]
[numthreads(__threads__, 1, 1)]
void CSMain(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID)
{
    // [thread_extent] blockIdx.x = __greater_blocks__
    // [thread_extent] threadIdx.x = __threads__
    uint bigger_block_id = gid.x;
    uint element_id = tid.x;
    uint largest = __largest__;

    for(uint smaller_block_id = 0; smaller_block_id < __smaller_blocks__; smaller_block_id++)
    {
        uint thread_id = tid.x + smaller_block_id * __threads__;
        uint cur_i = thread_id_to_idx(bigger_block_id, thread_id, __axis_size__, __axis_stride__);

        if(thread_id < __axis_size__)
        {
            buf[element_id].val = input0[cur_i];
            buf[element_id].index = cur_i;
        }
        else
        {
            buf[element_id].val = __boundary_value__;
            buf[element_id].index = cur_i;
        }
        GroupMemoryBarrierWithGroupSync();

        for(uint merge_step = 1; merge_step <= __thread_max_element__; merge_step <<= 1)
        {
            for(uint sort_step = merge_step; sort_step > 0; sort_step >>= 1)
            {
                if(element_id < __thread_max_element__/2 && sort_step!=__thread_max_element__)
                        bitonic_sort(element_id, sort_step, merge_step, largest);
                GroupMemoryBarrierWithGroupSync();
            }
        }

        GroupMemoryBarrierWithGroupSync();

        if(thread_id < __axis_size__)
        {
            output2[cur_i] = buf[element_id].index;
        }
        AllMemoryBarrierWithGroupSync();
    }

    // Per request to use single block to reduce all data
    for(uint mega_step = 1; mega_step <= __max_mega_step__; mega_step <<= 1)
    {
        for(uint smaller_block_id = 0; smaller_block_id < __smaller_blocks__; smaller_block_id += 2 * mega_step)
        {
            uint local_thread_id = tid.x + smaller_block_id * __threads__;
            uint local_i = thread_id_to_idx(bigger_block_id, local_thread_id, __axis_size__, __axis_stride__);
            if(local_thread_id < __axis_size__)
            {
                buf[element_id].index = output2[local_i];
                buf[element_id].val = input0[buf[element_id].index];
            }
            else
            {
                buf[element_id].val = __boundary_value__;
                buf[element_id].index = local_i;
            }
            GroupMemoryBarrierWithGroupSync();

            uint next_thread_id = local_thread_id + mega_step * __threads__;
            uint next_i = thread_id_to_idx(bigger_block_id, next_thread_id, __axis_size__, __axis_stride__);
            if(next_thread_id < __axis_size__)
            {
                buf[ - element_id - 1 + 2 * __thread_max_element__].index = output2[next_i];
                buf[ - element_id - 1 + 2 * __thread_max_element__].val = input0[buf[ - element_id - 1 + 2 * __thread_max_element__].index];
            }
            else
            {
                buf[ - element_id - 1 + 2 * __thread_max_element__].val = __boundary_value__;
                buf[ - element_id - 1 + 2 * __thread_max_element__].index = next_thread_id;
            }
            GroupMemoryBarrierWithGroupSync();

            for(uint sort_step = __thread_max_element__; sort_step > 0; sort_step >>= 1)
            {
                bitonic_sort(element_id, sort_step, __thread_max_element__ * 2, largest);
                GroupMemoryBarrierWithGroupSync();
            }
            GroupMemoryBarrierWithGroupSync();

            if(local_thread_id < __axis_size__)
            {
                output2[local_i] = buf[element_id].index;
            }
            AllMemoryBarrierWithGroupSync();
        }
    }

    AllMemoryBarrierWithGroupSync();
    if(element_id < __K__)
    {
        uint cur_i = thread_id_to_idx(bigger_block_id, element_id, __axis_size__, __axis_stride__);
        uint ans_i = thread_id_to_idx(bigger_block_id, element_id, __K__, __axis_stride__);
        uint index = output2[cur_i];
        uint local_tid = idx_to_thread_id(bigger_block_id, index, __axis_size__, __axis_stride__);
        output0[ans_i] = input0[index];
        output1[ans_i] = local_tid;
    }
}