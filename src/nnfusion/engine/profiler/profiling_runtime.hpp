// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * \brief Basic Datastructure used in profiling
 * \author wenxh
 */
#pragma once

#include <algorithm>
#include <memory>
#include <string>

#include <pwd.h>
#include <sqlite3.h>
#include <sys/types.h>
#include <unistd.h>

#include "nnfusion/common/type/data_buffer.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace std;
using namespace nnfusion;
using namespace nnfusion::graph;

#ifdef WIN32
#include <windows.h>
#define CLOSE_LIBRARY(a) FreeLibrary(a)
#define DLSYM(a, b) GetProcAddress(a, b)
#define DLIB_SUFFIX ".dll"
#define DL_HANDLE HMODULE
#else
#include <dlfcn.h>
#define CLOSE_LIBRARY(a) dlclose(a)
#define DLSYM(a, b) dlsym(a, b)
#define DLIB_SUFFIX ".so"
#define DL_HANDLE void*
#endif

namespace nnfusion
{
    namespace profiler

    {
        ///\bief Use this to store the result or other profiling details.
        struct ProfilingResult
        {
        private:
            vector<double> device_duration;
            vector<double> host_duration;
            bool ready = false;

        public:
            bool is_ready() const { return ready; }
            void set_ready() { ready = true; }
            //Get average time cost of host.
            double get_host_avg()
            {
                return host_duration.empty()
                           ? 0.0
                           : std::accumulate(host_duration.begin(), host_duration.end(), 0.0) /
                                 host_duration.size();
            }
            //Get average time cost inside the runtime.
            double get_device_avg()
            {
                return device_duration.empty()
                           ? 0.0
                           : std::accumulate(device_duration.begin(), device_duration.end(), 0.0) /
                                 device_duration.size();
            }

            const vector<double>& get_device_durations() { return device_duration; }
            const vector<double>& get_host_durations() { return host_duration; }
            void reset()
            {
                device_duration.clear();
                host_duration.clear();
            }

            void record_device_duration(double du) { device_duration.push_back(du); }
            void record_host_duration(double du) { host_duration.push_back(du); }
            using Pointer = shared_ptr<ProfilingResult>;
        };

        ///\brief The cache to store the time cost data, this should be connected
        // to some database tool.
        struct ProfilingCache
        {
            using KernelEmitter = nnfusion::kernels::KernelEmitter;

            struct KernelItem
            {
                std::string key_op;
                double cost;
            };

            template <typename F1>
            static std::string get_key_code(F1 emitter)
            {
                auto fu = emitter->get_or_emit_source();
                return fu->body_unit->get_code(); // TODO: need to be more specific
            }

            template <typename F1>
            static std::string get_key_op(F1 emitter)
            {
                auto& ctx = emitter->m_context;
                auto generic_op =
                    dynamic_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr());
                if (generic_op == nullptr)
                    return "";
                auto root = generic_op->localOpConfig.j_attrs;
                std::vector<std::vector<ssize_t>> tensor_shapes;
                std::vector<std::string> tensor_types;
                nnfusion::op::OpConfig::any inputs, outputs;

                for (int i = 0; i < ctx->inputs.size(); ++i)
                {
                    std::vector<ssize_t> tensor_shape;
                    auto& shape = ctx->inputs[i]->get_shape();
                    for (int j = 0; j < shape.size(); ++j)
                        tensor_shape.push_back(shape[j]);
                    tensor_shapes.push_back(std::move(tensor_shape));
                }
                root["input_shapes"] = std::move(tensor_shapes);
                tensor_shapes.clear();

                for (int i = 0; i < ctx->outputs.size(); ++i)
                {
                    std::vector<ssize_t> tensor_shape;
                    auto& shape = ctx->outputs[i]->get_shape();
                    for (int j = 0; j < shape.size(); ++j)
                        tensor_shape.push_back(shape[j]);
                    tensor_shapes.push_back(std::move(tensor_shape));
                }
                root["output_shapes"] = std::move(tensor_shapes);
                tensor_shapes.clear();

                for (int i = 0; i < ctx->dtypes.size(); ++i)
                {
                    tensor_types.push_back(ctx->dtypes[i]);
                }
                root["tensor_types"] = std::move(tensor_types);
                tensor_types.clear();

                return root.dump();
            }

            template <typename F1, typename F2>
            static double profile_timing_result(F1 ke, F2 func, string device_type)
            {
                auto emitter = ke->kernel;
                if (!ke->using_cache)
                    return func();

                static sqlite3* sqldb = NULL;
                if (!sqldb)
                {
                    NNFUSION_CHECK(SQLITE_OK ==
                                   sqlite3_open((getpwuid(getuid())->pw_dir +
                                                 std::string("/.cache/nnfusion_cache.db"))
                                                    .c_str(),
                                                &sqldb));
                    const char* table_create = R"(
CREATE TABLE IF NOT EXISTS KernelCache(
  key_code TEXT PRIMARY KEY NOT NULL,
  key_op TEXT NOT NULL,
  device_type TEXT NOT NULL,
  cost REAL );
)";
                    NNFUSION_CHECK(SQLITE_OK == sqlite3_exec(sqldb, table_create, NULL, 0, NULL));
                }

                auto key_code = get_key_code(emitter);
                if (key_code.size() == 0)
                {
                    NNFUSION_LOG(NNFUSION_WARNING)
                        << "Kernel `" << emitter->m_context->gnode->get_unique_name() << ":"
                        << emitter->m_context->gnode->get_name()
                        << "` is based on V1 kernel implementation, not supporting "
                           "caching for function_body!";
                    return func();
                }

                sqlite3_stmt* pStmt;

                const char* table_query_body = R"(
SELECT cost FROM KernelCache WHERE key_code = ?;
)";
                NNFUSION_CHECK(SQLITE_OK ==
                               sqlite3_prepare(sqldb, table_query_body, -1, &pStmt, 0));
                sqlite3_bind_text(pStmt, 1, key_code.data(), key_code.size(), SQLITE_STATIC);
                if (SQLITE_DONE != sqlite3_step(pStmt))
                {
                    double cost = sqlite3_column_double(pStmt, 0);
                    NNFUSION_CHECK(SQLITE_DONE == sqlite3_step(pStmt));
                    NNFUSION_CHECK(SQLITE_OK == sqlite3_finalize(pStmt));
                    NNFUSION_LOG(INFO) << device_type << "/" << emitter->get_function_name()
                                       << ": Using cached kernel time cost = " << cost;
                    return cost;
                }
                double result = func();
                auto key_op = get_key_op(emitter);
                NNFUSION_LOG(INFO) << device_type << "/" << emitter->get_function_name()
                                   << ": Updated cached kernel time cost = " << result;
                // NNFUSION_LOG(INFO) << get_key_op(emitter) << "\n" << key_code;

                const char* table_insert = R"(
INSERT INTO KernelCache(key_code, key_op, device_type, cost) VALUES(?, ?, ?, ?);
)";
                NNFUSION_CHECK(SQLITE_OK == sqlite3_prepare(sqldb, table_insert, -1, &pStmt, 0));
                sqlite3_bind_text(pStmt, 1, key_code.data(), key_code.size(), SQLITE_STATIC);
                sqlite3_bind_text(pStmt, 2, key_op.data(), key_op.size(), SQLITE_STATIC);
                sqlite3_bind_text(pStmt, 3, device_type.data(), device_type.size(), SQLITE_STATIC);
                sqlite3_bind_double(pStmt, 4, result);
                NNFUSION_CHECK(SQLITE_DONE == sqlite3_step(pStmt));
                NNFUSION_CHECK(SQLITE_OK == sqlite3_finalize(pStmt));
                return result;
            }
        };

        ///\brief Use this to manage the memory of kernels.
        class KernelMemory
        {
        public:
            using Pointer = unique_ptr<KernelMemory>;
            KernelMemory(kernels::KernelContext::Pointer kctx)
            {
                this->kctx = kctx;
                raw_input.clear();
                raw_output.clear();

                for (auto t : kctx->inputs)
                {
                    shared_ptr<char> i(new char[t->size()], [](char* p) { delete[] p; });
                    raw_input.push_back(move(i));
                }
                for (auto& t : kctx->outputs)
                {
                    shared_ptr<char> i(new char[t->size()], [](char* p) { delete[] p; });
                    raw_output.push_back(move(i));
                };
            }

            const KernelMemory& forward(int output_id, KernelMemory::Pointer& km, int input_id)
            {
                if (output_id >= raw_output.size() || input_id >= km->raw_input.size())
                {
                    NNFUSION_LOG(NNFUSION_WARNING) << "Invalid forward function.";
                    return *this;
                }
                km->raw_input[input_id] = raw_output[output_id];
                return *this;
            }

            bool load_input_from(int input_id, const void* data, size_t size)
            {
                auto buffsize = kctx->inputs[input_id]->size();
                // Check if the buffer is same size;
                if (input_id >= kctx->inputs.size() || size != buffsize)
                {
                    NNFUSION_LOG(ERROR) << "Input data size and memory buffer size don't match:";
                    return false;
                }
                auto status = memcpy(unsafe_input(input_id), data, buffsize);

                if (status == nullptr)
                {
                    NNFUSION_LOG(ERROR) << "Memcpy failed.";
                    return false;
                }
                return true;
            }

            bool set_output_from(int output_id, const void* data, size_t size)
            {
                auto buffsize = kctx->outputs[output_id]->size();
                // Check if the buffer is same size;
                if (output_id >= kctx->outputs.size() || size != buffsize)
                {
                    NNFUSION_LOG(ERROR) << "Input data size and memory buffer size don't match:";
                    return false;
                }
                auto status = memcpy(unsafe_output(output_id), data, buffsize);

                if (status == nullptr)
                {
                    NNFUSION_LOG(ERROR) << "Memcpy failed.";
                    return false;
                }
                return true;
            }

            template <typename T>
            bool load_input_from(const vector<T>& data, int input_id)
            {
                auto buffsize = kctx->inputs[input_id]->size();
                // Check if the buffer is same size;
                if (input_id >= kctx->inputs.size() || sizeof(T) * data.size() != buffsize)
                {
                    NNFUSION_LOG(ERROR) << "Input data size and memory buffer size don't match:";
                    return false;
                }
                auto status = memcpy(unsafe_input(input_id), (void*)data.data(), buffsize);

                if (status == nullptr)
                {
                    NNFUSION_LOG(ERROR) << "Memcpy failed.";
                    return false;
                }

                return true;
            }

            bool load_input_from(const DataBuffer& data, int input_id)
            {
                auto buffsize = kctx->inputs[input_id]->size();
                // Check if the buffer is same size;
                if (input_id >= kctx->inputs.size() || data.size_in_bytes() != buffsize)
                {
                    NNFUSION_LOG(ERROR) << "Input data size and memory buffer size don't match:"
                                        << data.size_in_bytes() << " != " << buffsize;
                    return false;
                }
                data.dump(unsafe_input(input_id));

                return true;
            }

            template <typename T>
            bool load_inputs(const vector<vector<T>>& data)
            {
                if (data.size() != kctx->inputs.size())
                {
                    NNFUSION_LOG(ERROR) << "Data items missmatch.";
                    return false;
                }
                for (int i = 0; i < data.size(); i++)
                {
                    if (load_input_from(data[i], i) == false)
                        return false;
                }
                return true;
            }

            bool load_inputs(const vector<DataBuffer>& data)
            {
                if (data.size() != kctx->inputs.size())
                {
                    NNFUSION_LOG(ERROR) << "Data items missmatch.";
                    return false;
                }
                for (int i = 0; i < data.size(); i++)
                {
                    if (load_input_from(data[i], i) == false)
                        return false;
                }
                return true;
            }

            template <typename T>
            vector<T> save_output(int output_id)
            {
                if (output_id > raw_output.size())
                {
                    NNFUSION_LOG(ERROR) << "Index exceeded the limit of vector.";
                    return vector<T>();
                }
                auto base = (T*)unsafe_output(output_id);
                auto buffsize = kctx->outputs[output_id]->size(false);
                vector<T> res(base, base + buffsize);
                return move(res);
            }

            DataBuffer save_output(int output_id, element::Type type)
            {
                if (output_id > raw_output.size())
                {
                    NNFUSION_LOG(ERROR) << "Index exceeded the limit of vector.";
                    return std::move(DataBuffer(type));
                }
                void* base = unsafe_output(output_id);
                size_t buffsize = kctx->outputs[output_id]->size(false);
                DataBuffer res(type);
                res.load(base, buffsize);
                return move(res);
            }

            template <typename T>
            vector<vector<T>> save_outputs()
            {
                vector<vector<T>> res;
                for (int i = 0; i < kctx->outputs.size(); i++)
                    res.push_back(save_output<T>(i));
                return res;
            }

            vector<DataBuffer> save_outputs(element::Type type)
            {
                vector<DataBuffer> res;
                for (int i = 0; i < kctx->outputs.size(); i++)
                    res.push_back(save_output(i, type));
                return move(res);
            }

            vector<DataBuffer> save_outputs(const vector<element::Type>& type)
            {
                NNFUSION_CHECK(type.size() == kctx->outputs.size()) << "Type vector size mismatch.";

                vector<DataBuffer> res;
                for (size_t i = 0; i < kctx->outputs.size(); i++)
                    res.push_back(save_output(i, type[i]));
                return move(res);
            }

            void* unsafe_input(int n)
            {
                if (n > raw_input.size())
                {
                    NNFUSION_LOG(ERROR) << "Index exceeded the limit of vector.";
                    return nullptr;
                }
                return raw_input[n].get();
            }

            void* unsafe_output(int n)
            {
                if (n > raw_output.size())
                {
                    NNFUSION_LOG(ERROR) << "Index exceeded the limit of vector.";
                    return nullptr;
                }
                return raw_output[n].get();
            }

            ///\brief At last, returned pointer shoule be translated into "T*[]" at runtime.
            ///\todo (wenxh)potential bug here, pointer may be used but deallocated.
            void** unsafe_inputs()
            {
                raw_inputs.reset(new char*[kctx->inputs.size()]);
                for (int i = 0; i < kctx->inputs.size(); i++)
                    raw_inputs.get()[i] = (char*)unsafe_input(i);
                return (void**)raw_inputs.get();
            }

            void** unsafe_outputs()
            {
                raw_outputs.reset(new char*[kctx->outputs.size()]);
                for (int i = 0; i < kctx->outputs.size(); i++)
                    raw_outputs.get()[i] = (char*)unsafe_output(i);
                return (void**)raw_outputs.get();
            }

        private:
            kernels::KernelContext::Pointer kctx;
            unique_ptr<char *> raw_inputs, raw_outputs;
            vector<shared_ptr<char>> raw_input, raw_output;
        };

        ///\brief The Context will have some basic info like:
        // -Input: Zeros, Ones, Randoms or Other Data.
        // -Output(optional): To check the output is right.
        // -Subject: Profile what subject.
        // -(Warmup)Times: .
        struct ProfilingContext
        {
        public:
            using Pointer = shared_ptr<ProfilingContext>;
            string working_dir = "profile/";
            size_t warmup_times = 5;
            size_t host_times = 1;
            size_t runtime_times = 100;
            // This emitter includes the kernel context;
            ProfilingResult result;
            kernels::KernelEmitter::Pointer kernel;
            ProfilingContext(kernels::KernelEmitter::Pointer kernel, bool using_cache = false)
                : using_cache(using_cache)
            {
                // this->using_cache = true;
                this->kernel = kernel;
                kernel_memory.reset(new KernelMemory(kernel->m_context));
            }
            ///\todo source code and function pointer need moved into cache;
            LanguageUnit_p source_code = nullptr;
            LanguageUnit_p cmake_code = nullptr;
            double (*entry_point)(void**, void**) = nullptr;
            ///\todo To be deprecated in future;
            // ProfilingContext(shared_ptr<ngraph::Node> node) { ; }
            KernelMemory::Pointer kernel_memory;
            bool using_cache;

            void reset()
            {
                source_code = nullptr;
                entry_point = nullptr;
                result.reset();
                // kernel_memory.release();
            }
        };

        ///\brief Restricted feature: Only support evaluation of result insteading of profiling.
        ///\todo (wenxh) support full-feature profiling, this to be done with new codegen.
        struct GraphEvaluationContext
        {
            shared_ptr<nnfusion::graph::Graph> graph = nullptr;
            GraphEvaluationContext(shared_ptr<nnfusion::graph::Graph> pGraph) { graph = pGraph; };
            void reset() { graph = nullptr; }
            ///\brief This function will generate a reference kernel for the GNode
            void set_profiling_context(shared_ptr<GNode> gnode, ProfilingContext::Pointer kctx)
            {
                // Need to check unique_name wether it works.
                if (prof_cache.find(gnode->get_unique_name()) == prof_cache.end())
                {
                    prof_cache[gnode->get_unique_name()] = kctx;
                }
            }

            ProfilingContext::Pointer get_profiling_context(shared_ptr<GNode> gnode)
            {
                if (prof_cache.find(gnode->get_unique_name()) != prof_cache.end())
                {
                    return prof_cache[gnode->get_unique_name()];
                }
                else
                {
                    NNFUSION_LOG(ERROR)
                        << "No valid Profiling Context for this node : " << gnode->get_name()
                        << " (op type : " << gnode->get_op_type() << ").";
                    return nullptr;
                }
            }

        private:
            ///\brief To store the output constant by the kernel.
            unordered_map<string, ProfilingContext::Pointer> prof_cache;
        };

        ///\brief The inteface for profiler runtime, which is binding to Device type.
        // Each device type should have one or more runtime.
        class IProfilingRuntime
        {
        public:
            ///\todo This interface is not safe, may access invlid memory address.
            bool execute(const ProfilingContext::Pointer& ke);
            virtual bool check_env() { return true; }
            double execute(const ProfilingContext::Pointer& ke, void** input, void** output);
            // Get the result of last run;
            using Pointer = shared_ptr<IProfilingRuntime>;
            string get_device_name() { return nnfusion::get_device_str(_dt); };
            NNFusion_DeviceType get_device_type() { return _dt; };
            virtual string get_name() { return get_device_name(); };
        private:
            virtual double invoke(const ProfilingContext::Pointer& ke, void** input, void** output);

        protected:
            NNFusion_DeviceType _dt;
            /*
            ///\todo To be provided in future, since we cannot use runtime api here.
            // We use Profiler class as Host here.
            virtual void* create_tensor(size_t bytes_size) = 0;
            virtual bool memcpyHtoD(void* host, void* device, size_t bytes_size) = 0;
            virtual bool memcpyDtoH(void* device, void* host, size_t bytes_size) = 0;
            */
        };
    }
}