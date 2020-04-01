// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include <assert.h>
#include <execinfo.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nnfusion/common/shape.hpp"
#include "nnfusion/common/type/element_type.hpp"
#include "nnfusion/common/util.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/operators/op.hpp"
#include "nnfusion/core/operators/op_define/abs.hpp"
#include "nnfusion/core/operators/op_define/acos.hpp"
#include "nnfusion/core/operators/op_define/add.hpp"
#include "nnfusion/core/operators/op_define/allreduce.hpp"
#include "nnfusion/core/operators/op_define/and.hpp"
#include "nnfusion/core/operators/op_define/argmax.hpp"
#include "nnfusion/core/operators/op_define/argmin.hpp"
#include "nnfusion/core/operators/op_define/asin.hpp"
#include "nnfusion/core/operators/op_define/atan.hpp"
#include "nnfusion/core/operators/op_define/avg_pool.hpp"
#include "nnfusion/core/operators/op_define/batch_norm.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/ceiling.hpp"
#include "nnfusion/core/operators/op_define/concat.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/core/operators/op_define/convert.hpp"
#include "nnfusion/core/operators/op_define/convolution.hpp"
#include "nnfusion/core/operators/op_define/cos.hpp"
#include "nnfusion/core/operators/op_define/cosh.hpp"
#include "nnfusion/core/operators/op_define/divide.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"
#include "nnfusion/core/operators/op_define/equal.hpp"
#include "nnfusion/core/operators/op_define/exp.hpp"
#include "nnfusion/core/operators/op_define/floor.hpp"
#include "nnfusion/core/operators/op_define/greater.hpp"
#include "nnfusion/core/operators/op_define/greater_eq.hpp"
#include "nnfusion/core/operators/op_define/less.hpp"
#include "nnfusion/core/operators/op_define/less_eq.hpp"
#include "nnfusion/core/operators/op_define/log.hpp"
#include "nnfusion/core/operators/op_define/lrn.hpp"
#include "nnfusion/core/operators/op_define/max.hpp"
#include "nnfusion/core/operators/op_define/max_pool.hpp"
#include "nnfusion/core/operators/op_define/maximum.hpp"
#include "nnfusion/core/operators/op_define/min.hpp"
#include "nnfusion/core/operators/op_define/minimum.hpp"
#include "nnfusion/core/operators/op_define/multiply.hpp"
#include "nnfusion/core/operators/op_define/negative.hpp"
#include "nnfusion/core/operators/op_define/not.hpp"
#include "nnfusion/core/operators/op_define/not_equal.hpp"
#include "nnfusion/core/operators/op_define/one_hot.hpp"
#include "nnfusion/core/operators/op_define/or.hpp"
#include "nnfusion/core/operators/op_define/pad.hpp"
#include "nnfusion/core/operators/op_define/parameter.hpp"
#include "nnfusion/core/operators/op_define/power.hpp"
#include "nnfusion/core/operators/op_define/product.hpp"
#include "nnfusion/core/operators/op_define/reduce.hpp"
#include "nnfusion/core/operators/op_define/reduce_window.hpp"
#include "nnfusion/core/operators/op_define/relu.hpp"
#include "nnfusion/core/operators/op_define/replace_slice.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/op_define/result.hpp"
#include "nnfusion/core/operators/op_define/reverse.hpp"
#include "nnfusion/core/operators/op_define/reverse_sequence.hpp"
#include "nnfusion/core/operators/op_define/select.hpp"
#include "nnfusion/core/operators/op_define/select_and_scatter.hpp"
#include "nnfusion/core/operators/op_define/sigmoid.hpp"
#include "nnfusion/core/operators/op_define/sign.hpp"
#include "nnfusion/core/operators/op_define/sin.hpp"
#include "nnfusion/core/operators/op_define/sinh.hpp"
#include "nnfusion/core/operators/op_define/slice.hpp"
#include "nnfusion/core/operators/op_define/softmax.hpp"
#include "nnfusion/core/operators/op_define/sqrt.hpp"
#include "nnfusion/core/operators/op_define/stop_gradient.hpp"
#include "nnfusion/core/operators/op_define/subtract.hpp"
#include "nnfusion/core/operators/op_define/sum.hpp"
#include "nnfusion/core/operators/op_define/tan.hpp"
#include "nnfusion/core/operators/op_define/tanh.hpp"
#include "nnfusion/core/operators/op_define/topk.hpp"

#define ktrace()                                                                                   \
    {                                                                                              \
        void* array[10];                                                                           \
        size_t size = backtrace(array, sizeof(array) / sizeof(*array));                            \
        char** strings = backtrace_symbols(array, size);                                           \
        if (NULL == strings)                                                                       \
        {                                                                                          \
            perror("backtrace_symbols");                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
        LOG(INFO) << " - Obtained " + size + " stack frames.";                                     \
        for (int i = 0; i < size; i++)                                                             \
            LOG(INFO) << "    # " + strings[i];                                                    \
        free(strings);                                                                             \
    }

namespace nnfusion
{
    namespace codegen
    {
        inline bool create_folder(std::string tar_path)
        {
            bool flag;
            int mkdir_status;
            struct stat s;
            int err = stat(tar_path.c_str(), &s);
            if (-1 == err)
            {
                mkdir_status = mkdir((tar_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                if (-1 == mkdir_status)
                {
                    LOG(INFO) << "Error creating directory: " + tar_path;
                    flag = false;
                }
                else
                    flag = true;
            }
            else
            {
                //LOG(INFO) << "Directory " << tar_path.c_str() << " already exists";
                flag = true;
            }
            return flag;
        }

        inline std::string get_file_from_templates(const std::string& rel_path)
        {
            static std::string abs_path;
            if (abs_path.size() == 0)
            {
                char exepath[1024];
                auto ret = readlink("/proc/self/exe", exepath, sizeof(exepath));
                CHECK(ret > 0);
                for (int i = strlen(exepath) - 1; i >= 0; --i)
                    if (exepath[i] == '/')
                    {
                        exepath[i] = 0;
                        break;
                    }
                abs_path = std::string(exepath) + "/templates/";
            }
            return abs_path + rel_path;
        }

        inline std::string get_content_from_templates(const std::string& rel_path)
        {
            std::ifstream in(get_file_from_templates(rel_path), std::ios::binary);
            std::string str((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            return str;
        }

        inline bool copy_file_from_templates(const std::string& rel_path,
                                             const std::string& target_name)
        {
            int at = 1, next;
            while (next = target_name.find('/', at), next >= 0)
            {
                create_folder(target_name.substr(0, next));
                at = next + 1;
            }
            std::ifstream in(get_file_from_templates(rel_path), std::ios::binary);
            std::ofstream out(target_name, std::ios::binary);
            out << in.rdbuf();
            return true;
        }
    } // namespace codegen
} // namespace nnfusion

using namespace std;
using namespace nnfusion;

#include "code_writer.hpp"
#include "device_type.hpp"
#include "gflags/gflags.h"
#include "nlohmann/json.hpp"
#include "nnfusion/util/util.hpp"
#include "type_info.hpp"

#define create_ptr(type, name, arg) shared_ptr<type> name(new type(arg))

// Uncomment this for quick debug
// #undef LOG(INFO)INFO
// #define LOG(INFO)INFO std::cout

// namespace nnfusion
// {
//     enum DeviceType
//     {
//         CUDA_GPU,
//         ROCM_GPU,
//         GENERIC_CPU
//     };
// }
