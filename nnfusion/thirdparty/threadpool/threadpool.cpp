// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "threadpool.h"

#include <cassert>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#else
#pragma warning(push)
#pragma warning(disable : 4267)
#endif
#include <unsupported/Eigen/CXX11/src/ThreadPool/Barrier.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif

using Eigen::Barrier;

namespace concurrency {
//
// ThreadPool
//
ThreadPool::ThreadPool(int num_threads, int numa_node)
    : numa_node_(numa_node) {
  impl_.reset(new Eigen::ThreadPoolTempl<NumaEnvironment>(num_threads, NumaEnvironment(numa_node)));
  device_.reset(new Eigen::ThreadPoolDevice(impl_.get(), impl_->NumThreads()));
}

void ThreadPool::Schedule(std::function<void()> fn) { 
  impl_->Schedule(fn);
}

void ThreadPool::ScheduleSync(std::function<void()> fn) {
  fn();
/*
  Barrier barrier(static_cast<unsigned int>(1));
  std::function<void()> fn_wrapper = [&barrier, &fn]() {
    fn();
    barrier.Notify();
  };
  impl_->Schedule([&fn_wrapper]() { fn_wrapper();});
  barrier.Wait();
*/
}

void ThreadPool::ParallelFor(int32_t total, std::function<void(int32_t)> fn) {
  if (total <= 0)
    return;

  if (total == 1)
  {
    fn(0);
    return;
  }

  // TODO: Eigen supports a more efficient ThreadPoolDevice mechanism
  // We will simply rely on the work queue and stealing in the short term.
  Barrier barrier(static_cast<unsigned int>(total));
  std::function<void(int32_t)> handle_iteration = [&barrier, &fn](int iteration) {
    fn(iteration);
    barrier.Notify();
  };

  for (int32_t id = 0; id < total; ++id) {
    Schedule([=, &handle_iteration]() { handle_iteration(id); });
  }

  barrier.Wait();
}

void ThreadPool::ParallelForRange(int64_t first, int64_t last, std::function<void(int64_t, int64_t)> fn) {
  if (last <= first) return;
  if (last - first == 1) {
    fn(first, last);
    return;
  }

  // TODO: Eigen supports a more efficient ThreadPoolDevice mechanism
  // We will simply rely on the work queue and stealing in the short term.
  Barrier barrier(static_cast<unsigned int>(last - first));
  std::function<void(int64_t, int64_t)> handle_range = [&barrier, &fn](int64_t first, int64_t last) {
    fn(first, last);
    barrier.Notify();
  };

  for (int64_t id = first + 1; id <= last; ++id) {
    Schedule([=, &handle_range]() { handle_range(id, id + 1); });
  }

  fn(first, first + 1);
  barrier.Wait();
}

int ThreadPool::NumThreads() const { return impl_->NumThreads(); }

int ThreadPool::CurrentThreadId() const { return impl_->CurrentThreadId(); }
}  // namespace concurrency
