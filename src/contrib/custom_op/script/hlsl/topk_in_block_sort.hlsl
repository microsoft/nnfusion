// Macros:
// __type__ : sorted type
// __axis_stride__ : stride for sorted axis 
// __max_element__ : Maximum element size for sorting
// __n__ : actual input element size for sorting
// __k__ : axis elements size
// __threads__ : thread in a block which is 

RWStructuredBuffer<__type__> input0: register(u0);
RWStructuredBuffer<__type__> output0: register(u1);
RWStructuredBuffer<int64_t> output1: register(u2);

__define_largest__

#ifdef LARGEST
#define CMP0 ((pos/step & 1) && (buf[pos].val > buf[pos+step].val)) || ((pos/step & 1) == 0) && (buf[pos].val < buf[pos+step].val)
#else
#define CMP0 ((pos/step & 1) && (buf[pos].val < buf[pos+step].val)) || ((pos/step & 1) == 0) && (buf[pos].val > buf[pos+step].val)
#endif

#ifdef LARGEST
#define CMP1 ((pos/gstep & 1) && buf[pos].val > buf[pos+step].val) || ((pos/gstep & 1) == 0 && (buf[pos].val < buf[pos+step].val))
#else
#define CMP1 buf[pos].val > buf[pos+step].val
#endif

struct type {
    __type__ val;
    int index;
};

groupshared type buf[__max_element__];

uint thread_id_to_idx(uint block_id, uint thread_id, uint axis_size)
{
    return (block_id / __axis_stride__) * (axis_size * __axis_stride__) + block_id % __axis_stride__ + thread_id * __axis_stride__;
}

// Bitonic Merging
// total: n
// partitian: n / (step * 2)
// each partitian: step threads
// total thread: n / 2
// task:
//  merge {pos, pos + step * 2}
//  {pos, pos + step - 1}, {pos + step, pos + step * 2 - 1}
void bitonic_merge(uint thread_id, uint step)
{
    uint pos = thread_id % step + (thread_id / step) * step * 2;
    if(CMP0)
    {
        type t =  buf[pos + step];
        buf[pos + step] = buf[pos];
        buf[pos] = t;
    }
}

// Bitonic Sorting
void bitonic_sort(uint thread_id, uint step, uint gstep)
{
    uint pos = thread_id % step + (thread_id / step) * step * 2;
    if(CMP1)
    {
        type t =  buf[pos + step];
        buf[pos + step] = buf[pos];
        buf[pos] = t;
    }
}

[numthreads(__threads__, 1, 1)] void TopK(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID)
{
    uint block_id = gid.x;
    uint thread_id = tid.x;

    for(int t = thread_id * 2; t < (thread_id + 1) * 2; t++)
    {
        uint cur_i = thread_id_to_idx(block_id, t, __n__);
        if(t < __n__)
        {
            buf[t].val = input0[cur_i];
            buf[t].index = t;
        }
        else 
        {
            // Fill with padding value
            buf[t].val = -1;
            buf[t].index = __n__;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for(uint merge_step = 1; merge_step <= __max_element__; merge_step <<= 1)
    {
        if(merge_step < __max_element__)
            bitonic_merge(thread_id, merge_step);
        GroupMemoryBarrierWithGroupSync();
        for(uint sort_step = merge_step>>1; sort_step > 0; sort_step >>= 1)
        {
            bitonic_sort(thread_id, sort_step, merge_step);
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // Write Back
    for(int t = thread_id * 2; t < __k__ && t < (thread_id + 1) * 2; t++)
    {
        uint new_i = thread_id_to_idx(block_id, t, __k__);
        output0[new_i] = buf[t].val;
        output1[new_i] = int64_t(buf[t].index);
    }
    GroupMemoryBarrierWithGroupSync();
}