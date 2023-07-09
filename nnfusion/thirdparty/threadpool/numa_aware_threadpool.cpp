// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "numa_aware_threadpool.h"
#include <assert.h>

namespace concurrency {

//
// NumaAwareThreadPool
//
NumaAwareThreadPool::NumaAwareThreadPool(int num_numa_node, int num_thread_per_node)
    : m_num_numa_node(num_numa_node),
      m_num_thread_per_node(num_thread_per_node) {
  assert(m_num_numa_node > 0);
  assert(m_num_thread_per_node >= 0);

  if (1 == m_num_numa_node) {
    if (m_num_thread_per_node == 0) {
      m_num_thread_per_node = std::thread::hardware_concurrency();
    }
    std::unique_ptr<ThreadPool> threadpool(new ThreadPool(m_num_thread_per_node));
    m_threadpools.push_back(std::move(threadpool));
  }
  else {
    if (m_num_thread_per_node == 0) {
      m_num_thread_per_node = std::thread::hardware_concurrency() / m_num_numa_node;
    }

    for (int i = 0; i < m_num_numa_node; ++i) {
      std::unique_ptr<ThreadPool> threadpool(new ThreadPool(m_num_thread_per_node, i));
      m_threadpools.push_back(std::move(threadpool));
    }
  }
}

void NumaAwareThreadPool::Schedule(std::function<void()> fn, int numa_node) { 
  assert(numa_node < m_num_numa_node);

  m_threadpools[numa_node]->Schedule(fn);
}

void NumaAwareThreadPool::ScheduleSync(std::function<void()> fn, int numa_node) {
  assert(numa_node < m_num_numa_node);

  m_threadpools[numa_node]->ScheduleSync(fn);
}

void NumaAwareThreadPool::ParallelFor(int32_t total, std::function<void(int32_t)> fn, int numa_node) {
  assert(numa_node < m_num_numa_node);

  m_threadpools[numa_node]->ParallelFor(total, fn);
}

void NumaAwareThreadPool::ParallelForRange(int64_t first, int64_t last, std::function<void(int64_t, int64_t)> fn, int numa_node) {
  assert(numa_node < m_num_numa_node);

  m_threadpools[numa_node]->ParallelForRange(first, last, fn);
}

ThreadPool* NumaAwareThreadPool::GetRawThreadPool(int numa_node) {
  assert(numa_node < m_num_numa_node);

  return m_threadpools[numa_node].get();
}

int NumaAwareThreadPool::NumNumaNodes() const { return m_num_numa_node; }

}  // namespace concurrency
