// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "threadpool.h"

namespace concurrency {
/**
 * Generic class for instantiating thread pools.
 * Don't put any object of this type into a global variable in a Win32 DLL.
 */
class NumaAwareThreadPool {
 public:
  /*
  Initializes thread pool(s) given the number of NUMA nodes.
  */
  NumaAwareThreadPool(int num_numa_node=1, int num_thread_per_node=0);

  /*
  Enqueue a unit of work.
  */
  void Schedule(std::function<void()> fn, int numa_node=0);

  /*
  Enqueue a unit of work and wait for execution.
  */
  void ScheduleSync(std::function<void()> fn, int numa_node=0);

  /*
  Schedule work in the interval [0, total).
  */
  void ParallelFor(int32_t total, std::function<void(int32_t)> fn, int numa_node=0);

  /*
  Schedule work in the interval [first, last].
  */
  void ParallelForRange(int64_t first, int64_t last, std::function<void(int64_t, int64_t)> fn, int numa_node=0);

  ThreadPool* GetRawThreadPool(int numa_node=0);

  int NumNumaNodes() const;

 private:
  std::vector<std::unique_ptr<ThreadPool>> m_threadpools;
  int m_num_numa_node;
  int m_num_thread_per_node;
};

}  // namespace concurrency
