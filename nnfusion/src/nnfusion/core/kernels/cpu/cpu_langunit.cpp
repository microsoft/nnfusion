// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cpu_langunit.hpp"

using namespace nnfusion::kernels;

// Header
LU_DEFINE(header::thread, "#include <thread>\n");
LU_DEFINE(header::eigen_tensor,
          "#define EIGEN_USE_THREADS\n#include <unsupported/Eigen/CXX11/Tensor>\n");
LU_DEFINE(header::eigen_utils, "#include \"eigen_utils.hpp\"\n");
LU_DEFINE(header::eigen_spatial_convolution, "#include \"eigen_spatial_convolutions.h\"\n");
LU_DEFINE(header::cblas, "#include \"mkl_cblas.h\"\n#define CBLAS_ENUM_DEFINED_H\n");
LU_DEFINE(header::mlas, "#include \"mlas.h\"\n");
LU_DEFINE(header::threadpool, "#include \"numa_aware_threadpool.h\"\n");
LU_DEFINE(header::barrier, "#include \"barrier.h\"\n");
LU_DEFINE(header::simd, "#include <immintrin.h>\n");

// Macro

// Declaration
LU_DEFINE(declaration::eigen_global_thread_pool, "Eigen::ThreadPool *global_thread_pool;\n");
LU_DEFINE(declaration::eigen_global_thread_pool_device,
          "Eigen::ThreadPoolDevice *global_thread_pool_device;\n");
LU_DEFINE(declaration::worker_thread_pool,
          "concurrency::NumaAwareThreadPool *worker_thread_pool;\n")
LU_DEFINE(declaration::schedule_thread_pool,
          "concurrency::NumaAwareThreadPool *schedule_thread_pool;\n")
LU_DEFINE(declaration::superscaler_schedule_thread,
          "concurrency::NumaAwareThreadPool *superscaler_schedule_thread;\n")