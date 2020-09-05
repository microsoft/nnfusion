#pragma once

namespace concurrency {
  bool HaveHWLocTopology();

  // Returns true iff NUMA functions are supported.
  bool NUMAEnabled();

  // Returns the number of NUMA nodes present with respect to CPU operations.
  // Typically this will be the number of sockets where some RAM has greater
  // affinity with one socket than another.
  int NUMANumNodes();

  static const int kNUMANoAffinity = -1;

  // If possible sets affinity of the current thread to the specified NUMA node.
  // If node == kNUMANoAffinity removes affinity to any particular node.
  void NUMASetThreadNodeAffinity(int node);

  // Returns NUMA node affinity of the current thread, kNUMANoAffinity if none.
  int NUMAGetThreadNodeAffinity();
};
