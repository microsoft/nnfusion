StructuredBuffer<__value_type__> input0: register(t0);
RWStructuredBuffer<__value_type__> output0: register(u0);
RWStructuredBuffer<__index_type__> output1: register(u1);
// globallycoherent : This storage class causes memory barriers and syncs to flush data across the entire GPU such that other groups can see writes.
// Without this specifier, a memory barrier or sync will only flush a UAV within the current group.
globallycoherent RWStructuredBuffer<int> output2: register(u2);

struct type {
    __value_type__ val;
    int index;
};

groupshared type buf[__block_max_element__];

uint thread_id_to_idx(uint block_id, uint thread_id, uint axis_size, uint axis_stride)
{
    return (block_id / axis_stride) * (axis_size * axis_stride) + block_id % axis_stride + thread_id * axis_stride;
}

uint idx_to_thread_id(uint block_id, uint idx, uint axis_size, uint axis_stride)
{
    // return (block_id / axis_stride) * (axis_size * axis_stride) + block_id % axis_stride + thread_id * axis_stride;
    return (idx - (block_id / axis_stride) * (axis_size * axis_stride) - block_id % axis_stride) / axis_stride;
}

// Bitonic Merging
void bitonic_merge(uint element_id, uint step, uint largest)
{
    uint pos = element_id % step + (element_id / step) * step * 2;
    if((((pos/step & 1) == largest) && (buf[pos].val > buf[pos+step].val)) || (((pos/step & 1) == (largest^1)) && (buf[pos].val < buf[pos+step].val)))
    {
        type t =  buf[pos + step];
        buf[pos + step] = buf[pos];
        buf[pos] = t;
    }
}

// Bitonic Sorting
void bitonic_sort(uint element_id, uint step, uint gstep, uint largest)
{
    uint pos = element_id % step + (element_id / step) * step * 2;
    if((((pos/gstep & 1) == largest) && buf[pos].val > buf[pos+step].val) || (((pos/gstep & 1) == (largest^1) && (buf[pos].val < buf[pos+step].val))))
    {
        type t =  buf[pos + step];
        buf[pos + step] = buf[pos];
        buf[pos] = t;
    }
}

[numthreads(1, 1, 1)] void CSMain(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID)
{
    uint bigger_block_id = gid.x;
    uint smaller_block_id = gid.y;
    uint thread_id = tid.x + smaller_block_id * __threads__;
    uint element_id = tid.x;
    uint cur_i = thread_id_to_idx(bigger_block_id, thread_id, __axis_size__, __axis_stride__);
    uint ans_i = thread_id_to_idx(bigger_block_id, thread_id, __K__, __axis_stride__);

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

    uint largest = __largest__^(smaller_block_id&1);
    if(element_id < __thread_max_element__ / 2 )
    {
        for(uint merge_step = 1; merge_step <= __thread_max_element__; merge_step <<= 1)
        {
            if(merge_step < __thread_max_element__)
                bitonic_merge(element_id, merge_step, largest);
            GroupMemoryBarrierWithGroupSync();
            for(uint sort_step = merge_step>>1; sort_step > 0; sort_step >>= 1)
            {
                bitonic_sort(element_id, sort_step, merge_step, largest);
                GroupMemoryBarrierWithGroupSync();
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if(thread_id < __axis_size__)
    {
        output2[cur_i] = buf[element_id].index;
    }
    DeviceMemoryBarrierWithGroupSync();

    for(uint mega_step = 1; mega_step <= __max_mega_step__; mega_step <<= 1)
    {
        if(smaller_block_id % (2 * mega_step) == 0)
        {
            if(thread_id < __axis_size__)
            {
                buf[element_id].index = output2[cur_i];
                buf[element_id].val = input0[buf[element_id].index];
                //output2[cur_i] = mega_step;
            }
            else
            {
                buf[element_id].val = __boundary_value__;
                buf[element_id].index = cur_i;
            }
            // This is magic!
            GroupMemoryBarrierWithGroupSync();

            uint next_thread_id = thread_id + mega_step * __threads__;
            uint next_i = thread_id_to_idx(bigger_block_id, next_thread_id, __axis_size__, __axis_stride__);
            if(next_thread_id < __axis_size__)
            {
                buf[element_id + __thread_max_element__].index = output2[next_i];
                buf[element_id + __thread_max_element__].val = input0[buf[element_id + __thread_max_element__].index];
                //output2[next_i] = mega_step;
            }
            else
            {
                buf[element_id + __thread_max_element__].val = __boundary_value__;
                buf[element_id + __thread_max_element__].index = next_i;
            }
            GroupMemoryBarrierWithGroupSync();

            uint largest = __largest__^((smaller_block_id>>mega_step)&1);
            for(uint merge_step = 1; merge_step <= __thread_max_element__ * 2 ; merge_step <<= 1)
            {
                if(merge_step < __thread_max_element__ * 2)
                    bitonic_merge(element_id, merge_step, largest);
                GroupMemoryBarrierWithGroupSync();
                for(uint sort_step = merge_step>>1; sort_step > 0; sort_step >>= 1)
                {
                    bitonic_sort(element_id, sort_step, merge_step, largest);
                    GroupMemoryBarrierWithGroupSync();
                }
            }

            if(thread_id < __axis_size__)
            {
                uint right_half = largest ^ 1;
                output2[cur_i] = buf[element_id + __thread_max_element__ * right_half].index;
            }
            // This one is also dark magic!!!
            AllMemoryBarrierWithGroupSync();
        }
        AllMemoryBarrierWithGroupSync();
    }
    AllMemoryBarrierWithGroupSync();

    if(thread_id < __K__)
    {
        uint index = output2[cur_i];
        uint local_tid = idx_to_thread_id(bigger_block_id, index, __axis_size__, __axis_stride__);
        output0[ans_i] = input0[index];
        output1[ans_i] = local_tid;
    }
    AllMemoryBarrierWithGroupSync();
}