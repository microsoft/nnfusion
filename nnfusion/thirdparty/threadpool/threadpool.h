// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include "util.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#else
#pragma warning(push)
#pragma warning(disable : 4267)
#endif
#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif

namespace concurrency {

struct NumaEnvironment {
  int numa_node_;
  bool use_numa_;

  NumaEnvironment(int numa_node = kNUMANoAffinity)
      : numa_node_(numa_node),
        use_numa_(false)
  {
    if (numa_node_ != kNUMANoAffinity)
    {
      if (HaveHWLocTopology())
      {
        use_numa_ = true;
      }
    }
  }

  struct Task
  {
    std::function<void()> f;
  };

  // EnvThread constructor must start the thread,
  // destructor must join the thread.
  class EnvThread {
   public:
    EnvThread(std::function<void()> f) : thr_(std::move(f)) {}
    ~EnvThread() { thr_.join(); }
    // This function is called when the threadpool is cancelled.
    void OnCancel() { }

   private:
    std::thread thr_;
  };

  EnvThread* StartThread(std::function<void()> fn)
  {
    return new EnvThread(std::move(fn));
  }

  EnvThread* CreateThread(std::function<void()> f)
  {

    return StartThread([=]()
    {
      if (use_numa_)
      {
        NUMASetThreadNodeAffinity(numa_node_);
      }
      f();
    });
  }

  Task CreateTask(std::function<void()> f) { return Task{std::move(f)}; }
  void ExecuteTask(const Task& t) { t.f(); }
};

/**
 * Generic class for instantiating thread pools.
 * Don't put any object of this type into a global variable in a Win32 DLL.
 */
class ThreadPool {
 public:
  /*
  Initializes a thread pool given the current environment.
  */
  ThreadPool(int num_threads, int numa_node=kNUMANoAffinity);

  /*
  Enqueue a unit of work.
  */
  void Schedule(std::function<void()> fn);

  /*
  Enqueue a unit of work and wait for execution.
  */
  void ScheduleSync(std::function<void()> fn);

  /*
  Schedule work in the interval [0, total).
  */
  void ParallelFor(int32_t total, std::function<void(int32_t)> fn);

  /*
  Schedule work in the interval [first, last].
  */
  void ParallelForRange(int64_t first, int64_t last, std::function<void(int64_t, int64_t)> fn);

  // This is not supported until the latest Eigen
  // void SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions);

  int NumThreads() const;

  int CurrentThreadId() const;

  //Eigen::ThreadPool& GetHandler() { return impl_; }

  Eigen::ThreadPoolDevice* GetDevice() { return device_.get(); }

 private:
  std::unique_ptr<Eigen::ThreadPoolTempl<NumaEnvironment>> impl_;
  std::shared_ptr<Eigen::ThreadPoolDevice> device_;
  int numa_node_;
};

}  // namespace concurrency
