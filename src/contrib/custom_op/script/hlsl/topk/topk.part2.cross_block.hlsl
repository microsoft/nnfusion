// This function will be expaned to many dumplications like loop unroll since DX doesn't has global sync
[numthreads(1, 1, 1)] void cross_block_sort(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID)
{
    uint bigger_block_id = gid.x;
    uint smaller_block_id = gid.y;
    uint thread_id = SV_GroupThreadID.x + smaller_block_id * __threads__;
    uint element_id = SV_GroupThreadID.x;
    uint cur_i = thread_id_to_idx(bigger_block_id, thread_id, __axis_size__);
    uint largest = __largest__;

    // TBD: load data from output1, kept left half or right half
    for(uint merge_step = __thread_max_element__; merge_step < __axis_size__ * 2; merge_step <<= 1)
    {
        // Read data
        bitonic_merge(element_id, __thread_max_element__, largest);
        GroupMemoryBarrierWithGroupSync();
        for(uint sort_step = merge_step>>1; sort_step > 0; sort_step >>= 1)
        {
            bitonic_sort(element_id, sort_step, merge_step, largest);
            GroupMemoryBarrierWithGroupSync();
        }
        // Write back
    }
}