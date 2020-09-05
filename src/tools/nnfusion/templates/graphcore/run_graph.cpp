// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "picosha2.h"

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>

#include <popnn/BatchNorm.hpp>
#include <popnn/Pooling.hpp>
#include <popnn/SpatialSoftMax.hpp>
#include <popnn/codelets.hpp>

#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/Gather.hpp>
#include <popops/Pad.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>

#include <poputil/TileMapping.hpp>

poplar::Device device;
std::vector<poplar::ComputeSet> compsets;
poplar::program::Sequence prog;

const int NUM_TILES = 1216;

namespace
{
    // Return a HW device with the requested number of IPUs.
    // Exception is thrown if no devices with the requested
    // number are available.
    inline poplar::Device getIpuHwDevice(std::size_t numIpus)
    {
        auto dm = poplar::DeviceManager::createDeviceManager();
        auto hwDevices = dm.getDevices(poplar::TargetType::IPU, numIpus);
        if (hwDevices.size() > 0)
        {
            for (auto& d : hwDevices)
            {
                if (d.attach())
                {
                    return std::move(d);
                }
            }
        }
        throw std::runtime_error("No IPU hardware available.");
    }

    // Return an IPU Model device with the requested number of IPUs.
    inline poplar::Device getIpuModelDevice(std::size_t numIpus)
    {
        poplar::IPUModel ipuModel;
        ipuModel.numIPUs = numIpus;
        return ipuModel.createDevice();
    }

    void DEBUG(const char* notation)
    {
        static int use_debug = -1;
        if (use_debug < 0)
            use_debug = getenv("DEBUG") ? 1 : 0;
        if (use_debug)
            printf("[DEBUG] Preparing %s..\n", notation);
    }
}

void place_tensor(poplar::Graph& g, poplar::Tensor& tensor)
{
    return poputil::mapTensorLinearly(g, tensor);

    int num_cells = tensor.numElements(), per_tile, y;
    for (int i = NUM_TILES; i >= 1; --i)
        if (num_cells % i == 0)
        {
            y = i, per_tile = num_cells / i;
            break;
        }
    auto t = tensor.reshape({(size_t)y, (size_t)per_tile});
    for (int i = 0; i < y; ++i)
        g.setTileMapping(t[i], i);
}

template <class T>
poplar::Tensor load_constant(poplar::Graph& g,
                             std::vector<std::pair<std::string, void*>>& data_ptrs,
                             poplar::Type dtype,
                             std::vector<std::size_t> shapes,
                             const std::string& name,
                             bool ones_flag = false)
{
    if (shapes.size() == 0)
        shapes = {1};

    size_t tensor_size = 1LU;
    for (auto& shape : shapes)
        tensor_size *= shape;

    T* hptr = (T*)malloc(tensor_size * sizeof(T));
    if (ones_flag)
    {
        for (int i = 0; i < tensor_size; ++i)
            hptr[i] = 1;
    }
    else
    {
        std::ifstream t("Constant/" + name);
        std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        assert(str.size() == tensor_size * sizeof(T));

        memcpy(hptr, str.data(), str.size());
    }

    poplar::Tensor ret;
    if (getenv("USE_CONST"))
    {
        ret = g.addConstant<T>(dtype, shapes, hptr);
    }
    else
    {
        ret = g.addVariable(dtype, poplar::ArrayRef<std::size_t>(shapes), name);
        data_ptrs.push_back({"W_" + name, hptr});
        prog.add(poplar::program::Copy(g.addHostToDeviceFIFO("W_" + name, dtype, ret.numElements()),
                                       ret));
    }

    place_tensor(g, ret);
    return std::move(ret);
}

void print_tensor(const std::string& tensor_name, const poplar::Tensor& tensor)
{
    if (getenv("RESULT"))
        prog.add(poplar::program::PrintTensor(tensor_name, tensor));
}

poplar::Tensor compute_task(poplar::Graph& g,
                            const std::vector<poplar::Tensor>& inputs,
                            const std::string& autogen,
                            int step,
                            int offset,
                            int tails,
                            const std::vector<int>& shards)
{
    using namespace poplar;

    assert(shards.size() == inputs.size() + 1);
    static int disable_blockfusion = -1, normal_step = 0;
    if (disable_blockfusion < 0)
        disable_blockfusion = getenv("FUSE") ? 0 : 1;
    if (disable_blockfusion)
    {
        step = normal_step++;
        tails -= offset, offset = 0;
    }

    // Parse thread_extents
    std::vector<std::string> thread_names;
    std::vector<int> thread_extents;
    int idx = 0, next;
    while (next = autogen.find("// [thread_extent] threadIdx_", idx), next >= 0)
    {
        next += sizeof("// [thread_extent]");

        int tail = autogen.find(" = ", next);
        assert(tail >= 0);
        thread_names.push_back(autogen.substr(next, tail - next));
        thread_extents.push_back(atoi(autogen.c_str() + tail + 3));
        idx = tail + 1;
    }

    int tot_threads_per_shard = 1;
    for (int i = 0; i < thread_extents.size(); ++i)
        tot_threads_per_shard *= thread_extents[i];
    assert(offset + tot_threads_per_shard * shards.back() == tails);

    // Update BSP step
    assert(offset + tot_threads_per_shard * shards.back() <= NUM_TILES * 6);

    std::string cs_name = "compset-" + std::to_string(step);

    std::string kernel_name = "AntaresVertexCodeletsImpl_" + picosha2::hash256_hex_string(autogen);

    static std::unordered_set<std::string> added_kernel_names;

    printf("Loading 1 kernel with name: %s (tiles = %dx%d from compset %s, super-step = %d:%d)\n",
           kernel_name.c_str(),
           tot_threads_per_shard,
           shards.back(),
           cs_name.c_str(),
           step,
           offset);

    // No super-step decrease
    static int prev_step = -1;
    assert(prev_step <= step);
    prev_step = step;

    if (compsets.size() <= step)
    {
        assert(compsets.size() == step);
        compsets.push_back(g.addComputeSet(cs_name));
        prog.add(program::Execute(compsets.back()));
    }

    ComputeSet& compset = compsets[step];

    // Parse inputs & outputs
    std::vector<std::pair<std::string, std::vector<std::size_t>>> func_args;
    {
        idx = autogen.find("// [func_args]");
        assert(idx >= 0), idx += sizeof("// [func_args]");
        next = autogen.find("\n", idx);
        auto args_str = autogen.substr(idx, next - idx);
        idx = 0;
        while (next = args_str.find("({", idx), next >= 0)
        {
            std::string arg_def = args_str.substr(idx, next - idx);
            std::vector<std::size_t> arg_shape = {};
            for (int i = next + 2, j = i + 1; j < args_str.size(); ++j)
            {
                if (args_str[j] == ',' || args_str[j] == '}')
                {
                    arg_shape.push_back(atoi(args_str.c_str() + i));
                    i = j + 1;
                    if (args_str[j] == '}')
                        break;
                }
            }
            assert(arg_def.size() > 0);
            func_args.push_back({std::move(arg_def), std::move(arg_shape)});
            idx = args_str.find("; ", next);
            if (idx < 0)
                break;
            idx += sizeof("; ") - 1;
        }
    }

    std::stringstream s_src;
    s_src << R"(
#include <poplar/Vertex.hpp>

extern "C" float tanhf(float);
extern "C" float expf(float);
extern "C" float powf(float, float);
#define __expf expf

template <class T> inline T max(const T &x, const T &y) { return x > y ? x : y; }
template <class T> inline T min(const T &x, const T &y) { return x < y ? x : y; }

using namespace poplar;

class )" << kernel_name
          << R"(: public Vertex {
public:

  bool compute() {)";

    s_src << autogen;

    s_src << R"(
    return true;
  }

public:)";

    for (auto& it : thread_names)
        s_src << "\n  int " << it << ";";
    for (auto& it : func_args)
        s_src << "\n  " << it.first << ";";
    s_src << "\n}; // class Vertex";

    if (added_kernel_names.count(kernel_name) == 0)
    {
        g.addCodelets(s_src);
        added_kernel_names.insert(kernel_name);
    }

    std::sort(func_args.begin(),
              func_args.end(),
              [](const decltype(func_args[0])& x, const decltype(func_args[0])& y) {
                  return x.first < y.first;
              });

    auto get_name = [](const std::string& str_def) {
        int pos = str_def.find(' ');
        assert(pos >= 0);
        return str_def.substr(pos + 1);
    };

    auto get_type = [](const std::string& str_def) {
        int pos1 = str_def.find("Vector<"), pos2;
        assert(pos1 >= 0);
        pos1 += sizeof("Vector<") - 1;
        pos2 = str_def.find('>', pos1);
        auto str_type = str_def.substr(pos1, pos2 - pos1);
        if (str_type == "float")
            return FLOAT;
        if (str_type == "int")
            return INT;
        if (str_type == "half")
            return HALF;
        throw std::runtime_error(("Not recognized tensor type: " + str_type).c_str());
    };

    assert(func_args.size() >= 1);
    assert(inputs.size() + 1 == func_args.size());

    auto& output_arg = func_args[func_args.size() - 1];
    std::vector<std::size_t> output_shape_concat = {(size_t)shards.back()};
    for (auto& arg : output_arg.second)
        output_shape_concat.push_back(arg);

    Tensor result = g.addVariable(get_type(output_arg.first),
                                  poplar::ArrayRef<std::size_t>(output_shape_concat),
                                  kernel_name);

    auto tiled_result =
        result.reshape({(size_t)shards.back(),
                        (size_t)tot_threads_per_shard,
                        result.numElements() / (shards.back() * tot_threads_per_shard)});
    for (int sh = 0; sh < shards.back(); ++sh)
    {
        for (int curr_thread = 0; curr_thread < tot_threads_per_shard; ++curr_thread)
        {
            auto v = g.addVertex(compset, kernel_name);
            for (int i = thread_extents.size() - 1, ct = curr_thread; i >= 0;
                 ct /= thread_extents[i], --i)
                g.setInitialValue(v["threadIdx_" + std::to_string(i)], ct % thread_extents[i]);

            for (int i = 0; i + 1 < func_args.size(); ++i)
                g.connect(v[get_name(func_args[i].first)],
                          inputs[i].reshape({(size_t)shards[i],
                                             inputs[i].numElements() / shards[i]})[sh % shards[i]]);
            g.connect(v[get_name(output_arg.first)], tiled_result[sh][curr_thread]);
            int tile_mapped = (offset + (sh * tot_threads_per_shard + curr_thread)) % NUM_TILES;
            g.setTileMapping(v, tile_mapped);
            g.setTileMapping(tiled_result[sh][curr_thread], tile_mapped);

            if (int(device.getId()) < 0)
                g.setCycleEstimate(v, 20);
        }
    }

    offset += tot_threads_per_shard * shards.back();
    return std::move(result);
}

int main(int argc, char** argv)
{
    using namespace poplar;

    device = getenv("IPU") ? getIpuHwDevice(1) : getIpuModelDevice(1);
    printf("Ipu Device Id = %d\n", (int)device.getId());

    Graph g(device.getTarget());

    poplin::addCodelets(g);
    popnn::addCodelets(g);
    popops::addCodelets(g);

    std::vector<std::pair<std::string, void*>> data_ptrs;

#include "nnfusion_rt.h"

    printf("Ipu starts to build..\n");
    Engine engine(g, prog);
    for (int i = 0; i < data_ptrs.size(); ++i)
        engine.connectStream(data_ptrs[i].first, data_ptrs[i].second);
    engine.load(device);

    std::cout << "Running program\n";
    auto run = [&](int runs = 1) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; ++i)
            engine.run(0);
        auto end = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << std::fixed << std::setprecision(8) << "Program complete once in "
                  << ns * 1e-9 / runs << " sec.\n";
    };

    run(1);
    run(100);
    if (getenv("PROF"))
        engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});
    run(1);
    return EXIT_SUCCESS;
}
