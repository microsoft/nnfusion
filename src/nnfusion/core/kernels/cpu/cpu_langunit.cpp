// Microsoft (c) 2019, NNFusion Team
#include "cpu_langunit.hpp"

using namespace nnfusion::kernels;

// Header
LU_DEFINE(header::thread, "#include <thread>\n");
LU_DEFINE(header::eigen_tensor,
          "#define EIGEN_USE_THREADS\n#include <unsupported/Eigen/CXX11/Tensor>\n");
LU_DEFINE(header::eigen_utils, "#include \"eigen_utils.hpp\"\n");
LU_DEFINE(header::cblas, "#include \"cblas.h\"\n");
LU_DEFINE(header::mlas, "#include \"mlas/inc/mlas.h\"\n#include \"mlas/inc/threadpool.h\"\n");
LU_DEFINE(header::barrier, "#include \"barrier.h\"\n");

// Macro

// Declaration
LU_DEFINE(declaration::eigen_global_thread_pool, "Eigen::ThreadPool *global_thread_pool;\n");
LU_DEFINE(declaration::eigen_global_thread_pool_device,
          "Eigen::ThreadPoolDevice *global_thread_pool_device;\n");
LU_DEFINE(declaration::cblas_sgemm_batch,
          R"(extern "C" {
void cblas_sgemm_batch(const CBLAS_LAYOUT Layout,
        const CBLAS_TRANSPOSE* transa_array,
        const CBLAS_TRANSPOSE* transb_array,
        const int64_t* m_array,
        const int64_t* n_array,
        const int64_t* k_array,
        const float* alpha_array,
        const float** a_array,
        const int64_t* lda_array,
        const float** b_array,
        const int64_t* ldb_array,
        const float* beta_array,
        float** c_array,
        const int64_t* ldc_array,
        const int64_t group_count,
        const int64_t* group_size);
})")
LU_DEFINE(declaration::mlas_global_thread_pool, "MLAS_THREADPOOL *mlas_global_thread_pool;\n")
