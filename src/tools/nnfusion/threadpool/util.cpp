#define NOMINMAX
#include "util.h"
#include "hwloc.h"
#include <algorithm>

namespace concurrency {
static hwloc_topology_t hwloc_topology_handle;

bool HaveHWLocTopology() {
  // One time initialization
  static bool init = []() {
    if (hwloc_topology_init(&hwloc_topology_handle)) {
      //LOG(ERROR) << "Call to hwloc_topology_init() failed";
      return false;
    }
    if (hwloc_topology_load(hwloc_topology_handle)) {
      //LOG(ERROR) << "Call to hwloc_topology_load() failed";
      return false;
    }
    return true;
  }();
  return init;
}

// Return the first hwloc object of the given type whose os_index
// matches 'index'.
hwloc_obj_t GetHWLocTypeIndex(hwloc_obj_type_t tp, int index) {
  hwloc_obj_t obj = nullptr;
  if (index >= 0) {
    while ((obj = hwloc_get_next_obj_by_type(hwloc_topology_handle, tp, obj)) !=
           nullptr) {
      if (obj->os_index == index) break;
    }
  }
  return obj;
}

bool NUMAEnabled() { return (NUMANumNodes() > 1); }

int NUMANumNodes() {
  if (HaveHWLocTopology()) {
    int num_numanodes =
        hwloc_get_nbobjs_by_type(hwloc_topology_handle, HWLOC_OBJ_NUMANODE);
    return std::max(1, num_numanodes);
  } else {
    return 1;
  }
}

void NUMASetThreadNodeAffinity(int node) {
  // Find the corresponding NUMA node topology object.
  hwloc_obj_t obj = GetHWLocTypeIndex(HWLOC_OBJ_NUMANODE, node);
  if (obj) {
    hwloc_set_cpubind(hwloc_topology_handle, obj->cpuset,
                      HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT);
  } else {
    //LOG(ERROR) << "Could not find hwloc NUMA node " << node;
  }
}

int NUMAGetThreadNodeAffinity() {
  int node_index = kNUMANoAffinity;
  if (HaveHWLocTopology()) {
    hwloc_cpuset_t thread_cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(hwloc_topology_handle, thread_cpuset,
                      HWLOC_CPUBIND_THREAD);
    hwloc_obj_t obj = nullptr;
    // Return the first NUMA node whose cpuset is a (non-proper) superset of
    // that of the current thread.
    while ((obj = hwloc_get_next_obj_by_type(
                hwloc_topology_handle, HWLOC_OBJ_NUMANODE, obj)) != nullptr) {
      if (hwloc_bitmap_isincluded(thread_cpuset, obj->cpuset)) {
        node_index = obj->os_index;
        break;
      }
    }
    hwloc_bitmap_free(thread_cpuset);
  }
  return node_index;
}
}
