StructuredBuffer<__value_type__> input0: register(t0);
RWStructuredBuffer<__index_type__> output0: register(u0);

groupshared __value_type__ buf[__block_max_element__];

uint thread_id_to_idx(uint block_id, uint e_id, uint axis_size, uint axis_stride)
{
    return (block_id / axis_stride) * (axis_size * axis_stride) + block_id % axis_stride + e_id * axis_stride;
}

[RootSignature("DescriptorTable(SRV(t0, numDescriptors=1), UAV(u0, numDescriptors=1))")]
[numthreads(__threads__, 1, 1)]
void CSMain(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID)
{
    // [thread_extent] blockIdx.x = __blocks__
    // [thread_extent] threadIdx.x = __threads__
    uint block_id = gid.x;
    uint thread_id = tid.x;
    uint lane_count = WaveGetLaneCount();
    __index_type__ max_element_id = 0;
    __value_type__ max_block_value = __boundary_value__;

    for(uint step=0; step < __step_size__; step+=1024)
    {
        uint element_id = step + thread_id;
        uint cur_element_id = thread_id_to_idx(block_id, element_id, __axis_size__, __axis_stride__);
        if(element_id < __axis_size__)
            buf[thread_id] = input0[cur_element_id];
        else
            buf[thread_id] = __boundary_value__;
        __value_type__ lane_val = buf[thread_id];
        GroupMemoryBarrierWithGroupSync();
        // Loaded all data into register
        __value_type__ warp_max = WaveActiveMax(lane_val);
        if(lane_val == warp_max)
        {
            uint buf_id = thread_id / lane_count;
            buf[buf_id] = warp_max;
            buf[buf_id + lane_count] = element_id;
        }
        GroupMemoryBarrierWithGroupSync();
        // First wave:
        if(thread_id < lane_count)
        {
            if(thread_id > (__axis_size__ - 1 - step) / lane_count)
            {
                buf[thread_id] = __boundary_value__;
            }

            __value_type__ lane_val = buf[thread_id];
            __value_type__ warp_max = WaveActiveMax(lane_val);
            __index_type__ local_max_element_id = 0;
            if(lane_val == warp_max)
            {
                local_max_element_id = buf[thread_id + lane_count];
            }       
            __index_type__ warp_max_element_id = WaveActiveMax(local_max_element_id);

            // First thread
            if(WaveIsFirstLane())
            {
                if(warp_max > max_block_value)
                {
                    max_block_value = warp_max;
                    max_element_id = warp_max_element_id;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if(thread_id == 0)
    {
        output0[block_id] = max_element_id;
    }
}