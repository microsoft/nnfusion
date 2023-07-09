// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "d3dx12_nnfusion.h"

int main(int argc, char** argv)
{
    D3DDevice device(false, false);
    device.Init();

    using namespace nnfusion_dml;

#include "nnfusion_rt.h"

    auto evaluateQueue = [&](const std::vector<ID3D12CommandList*>& cmdQueue, const char* qtype) {
        std::chrono::high_resolution_clock::time_point t1 =
            std::chrono::high_resolution_clock::now();
        constexpr int NUM_STEPS = 10;
        for (int i = 0; i < NUM_STEPS; i++)
        {
            device.pCommandQueue->ExecuteCommandLists(cmdQueue.size(), cmdQueue.data());
            device.AwaitExecution();
        }
        std::chrono::high_resolution_clock::time_point t2 =
            std::chrono::high_resolution_clock::now();
        printf("DxCompute Time per Run for [%s] = %g sec.\n",
               qtype,
               std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() /
                   NUM_STEPS);
    };

    if (profCostDict.size() > 0)
    {
        double evaluate_sum = 0.0;
        std::multimap<double, std::wstring> orderedProf;
        for (auto& it : profCostDict)
        {
            double timecost = it.second.first * it.second.second;
            evaluate_sum += timecost;
            orderedProf.insert(std::make_pair(timecost, it.first));
        }
        for (auto it = orderedProf.rbegin(); it != orderedProf.rend(); ++it)
        {
            auto ratio = std::to_wstring(it->first * 1e2 / evaluate_sum);
            if (ratio.size() > 6)
                ratio = ratio.substr(0, 6);
            printf("%8ls%%  %6d  %4.8lf\t%ls\n",
                   ratio.c_str(),
                   profCostDict[it->second].second,
                   it->first,
                   it->second.c_str());
        }
        printf("DxCompute Time per Run for [Profile Sum] = %g sec.\n", evaluate_sum);
    }

    evaluateQueue(cmdQueue, "Standard Queue");
    printf("Total GPU Memory Allocated = %g MB\n", totalGPUMemoryAccess / double(1024 * 1024));
    system("pause");
    return 0;
}
