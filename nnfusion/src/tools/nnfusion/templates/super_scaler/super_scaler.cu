// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "super_scaler.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <queue>
#include <thread>

#include "mpi.h"
#include "nccl.h"
#include "unistd.h"

#define NNSCALER_MPICHECK(cmd)                                                                     \
    do                                                                                             \
    {                                                                                              \
        int e = cmd;                                                                               \
        if (e != MPI_SUCCESS)                                                                      \
        {                                                                                          \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define NNSCALER_CUDACHECK(cmd)                                                                    \
    do                                                                                             \
    {                                                                                              \
        cudaError_t e = cmd;                                                                       \
        if (e != cudaSuccess)                                                                      \
        {                                                                                          \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define NNSCALER_NCCLCHECK(cmd)                                                                    \
    do                                                                                             \
    {                                                                                              \
        ncclResult_t r = cmd;                                                                      \
        if (r != ncclSuccess)                                                                      \
        {                                                                                          \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

class TaskQueue
{
public:
    TaskQueue()
    {
        task_enable = true;
        monitor = std::thread(&TaskQueue::run, this);
    }

    void push(std::thread&& thread)
    {
        mtx.lock();
        threads.push(move(thread));
        mtx.unlock();
    }

    bool is_empty()
    {
        bool isempty;
        mtx.lock();
        isempty = threads.empty();
        mtx.unlock();
        return isempty;
    }

    void run()
    {
        while (task_enable)
        {
            if (!is_empty())
            {
                mtx.lock();
                auto& task = threads.front();
                mtx.unlock();
                task.join();
                threads.pop();
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }

    void start() { monitor.detach(); }
    void end() { task_enable = false; }
private:
    std::queue<std::thread> threads;
    std::mutex mtx;
    std::thread monitor;
    bool task_enable;
} task_queue;

// Context of super_scaler instance
static int super_scaler_myRank = -1;
static int super_scaler_nRanks = -1;
static int super_scaler_localRank = -1;
static int nDev = 1;
static cudaStream_t* stream;
static ncclUniqueId id;
static ncclComm_t* comm;

static uint32_t align_to_block_size(uint32_t threads, uint32_t block_size)
{
    if (threads > (1u << 31) - 1)
    {
        throw std::runtime_error("Cuda can't handle threads > 2^31 - 1.");
    }
    uint32_t r = (threads + block_size - 1) / block_size;
    return r;
}

static uint64_t super_scaler_getHostHash(const char* string)
{
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++)
    {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void super_scaler_getHostName(char* hostname, int maxlen)
{
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++)
    {
        if (hostname[i] == '.')
        {
            hostname[i] = '\0';
            return;
        }
    }
}

__global__ static void gradientsAverage(float* gradients, int size, int super_scaler_nRanks)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        gradients[tid] /= super_scaler_nRanks;
    }
}

void super_scaler_initialization()
{
    //initializing MPI
    NNSCALER_MPICHECK(MPI_Init(NULL, NULL));
    NNSCALER_MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &super_scaler_myRank));
    NNSCALER_MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &super_scaler_nRanks));

    //calculating super_scaler_localRank which is used in selecting a GPU
    uint64_t hostHashes[super_scaler_nRanks];
    char hostname[1024];
    super_scaler_getHostName(hostname, 1024);
    hostHashes[super_scaler_myRank] = super_scaler_getHostHash(hostname);
    NNSCALER_MPICHECK(MPI_Allgather(
        MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashes, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

    //printf("[SuperScaler:Info] TotalRank = %d; MyRank = %d; HostName = %s; HostHash = %u;\n", 
    //   super_scaler_nRanks, super_scaler_myRank, hostname, hostHashes[super_scaler_myRank]);
    auto dev_no = getenv("NNFUSION_DEV_NO");
    if(dev_no == nullptr)
    {
        super_scaler_localRank = 0;
        for (int p = 0; p < super_scaler_nRanks; p++)
        {
            if (p == super_scaler_myRank)
            {
                break;
            }
            if (hostHashes[p] == hostHashes[super_scaler_myRank])
            {
                super_scaler_localRank++;
            }
        }
    }
    else
    {
        super_scaler_localRank = atoi(dev_no);
    }

    int dev_cnt;
    cudaGetDeviceCount(&dev_cnt);
    if(super_scaler_localRank >= dev_cnt)
    {
        // printf("[SuperScaler:Warning] localRand is more than device count %d >= %d.\n",
        //      super_scaler_localRank, dev_cnt);
        super_scaler_localRank = 0;
    }

    // printf("[SuperScaler:Info] Choose device: %d.\n", super_scaler_localRank);
    NNSCALER_CUDACHECK(cudaSetDevice(super_scaler_localRank));

    // Stream per device;
    stream = new cudaStream_t;
    NNSCALER_CUDACHECK(cudaStreamCreate(stream));
    
    //generating NCCL unique ID at one process and broadcasting it to all
    if (super_scaler_myRank == 0)
        ncclGetUniqueId(&id);
    NNSCALER_MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    //initializing NCCL, group API is required around ncclCommInitRank as it is called across multiple GPUs in each thread/process
    comm = new ncclComm_t;
    NNSCALER_NCCLCHECK(ncclGroupStart());
    NNSCALER_NCCLCHECK(ncclCommInitRank(comm, super_scaler_nRanks, id, super_scaler_myRank));
    NNSCALER_NCCLCHECK(ncclGroupEnd());

    //todo: how to know how many processes issued and sync?
    task_queue.start();
}

void super_scaler_finalization()
{
    task_queue.end();
    MPI_Barrier(MPI_COMM_WORLD);
    //finalizing NCCL
    NNSCALER_NCCLCHECK(ncclCommDestroy(*comm));
    NNSCALER_CUDACHECK(cudaStreamDestroy(*stream));
    //finalizing MPI
    NNSCALER_MPICHECK(MPI_Finalize());
    delete comm;
    delete stream;
}

void super_scaler_sync()
{
    while (!task_queue.is_empty())
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    MPI_Barrier(MPI_COMM_WORLD);
}

void super_scaler_all_reduce(float* gradients,
                             float* out_gradients,
                             int size,
                             void* exestream,
                             void (*callback)(void*),
                             void* callback_context)
{
    cudaStream_t* run_on_stream = exestream==nullptr?stream:(cudaStream_t*)exestream;

    //get gradients after allreduce
    if(super_scaler_nRanks>1)
    {
        uint32_t block_size_x = 512;
        size_t block_cnt = align_to_block_size(size, block_size_x);
        //todo: Could this change device for main thread?
        gradientsAverage<<<block_cnt, block_size_x, 0, *run_on_stream>>>(
            gradients, size, super_scaler_nRanks);
    }

    //calling NCCL communication API. Group API is required when using multiple devices per thread/process
    NNSCALER_NCCLCHECK(ncclGroupStart());
    NNSCALER_NCCLCHECK(ncclAllReduce((const void*)gradients,
                                         (void*)out_gradients,
                                         size,
                                         ncclFloat,
                                         ncclSum,
                                         *comm,
                                         *run_on_stream));
    NNSCALER_NCCLCHECK(ncclGroupEnd());

    //call back
    if (callback != nullptr)
        (*callback)(callback_context);
}

/*
void super_scaler_all_reduce_async(float* gradients,
                                   float* out_gradients,
                                   int size,
                                    cudaStream_t* exestream,
                                   void (*callback)(void*),
                                   void* callback_context)
{
    std::thread NNSCALER_allreduce(
        super_scaler_all_reduce, gradients, out_gradients, size, exestream, callback, callback_context);
    task_queue.push(move(NNSCALER_allreduce));
}
*/

int super_scaler_get_localrank()
{
    return super_scaler_localRank;
}