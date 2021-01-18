// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cuda_helper.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion::kernels;
using CodeWriter = nnfusion::codegen::CodeWriter;

LanguageUnit_p cuda::get_math_kernel(const std::string& name,
                                     const std::string& math_kernel,
                                     const std::vector<std::string>& data_types)
{
    NNFUSION_CHECK(std::count(name.begin(), name.end(), '-') == 0);
    std::string mangled_name = "declaration::function_def_inline_" + name;
    // TODO: handle data_types containing underline, like long_long
    // Output type should be ignore
    for (size_t i = 0; i < data_types.size() - 1; i++)
    {
        mangled_name += "-" + data_types[i];
    }
    shared_ptr<LanguageUnit> cw(new LanguageUnit(mangled_name));
    auto& writer = *cw;
    if (math_kernel.size())
    {
        auto num_inputs = data_types.size() - 1;
        writer << "__device__ __forceinline__ " << data_types[num_inputs] << " " << name << "(";
        for (size_t i = 0; i < num_inputs - 1; i++)
        {
            writer << data_types[i] << " x" << i << ", ";
        }
        writer << data_types[num_inputs - 1] << " x" << num_inputs - 1;
        writer << ")\n";
        writer << "{\n";
        writer.indent++;
        {
            writer << "return " + math_kernel << ";\n";
        }
        writer.indent--;
        writer << "}\n";
    }
    return cw;
}

LanguageUnit_p cuda::get_atomic_math_kernel(const std::string& name,
                                            const std::string& math_kernel,
                                            const std::string data_type)
{
    NNFUSION_CHECK(std::count(name.begin(), name.end(), '-') == 0);
    std::string mangled_name = "declaration::function_def_atomic" + name;
    std::string code = "/*Empty Not Supposed*/\n";
    if (math_kernel.size())
    {
        code = nnfusion::op::create_code_from_template(
            R"(
__device__ void atomic_@name@(@T@* ptr, @T@ x1) {
  int* i_ptr = (int*) ptr;
  int i_x0 = *i_ptr;
  int assumed;
  do {
    assumed = i_x0;
    @T@ x0 = __int_as_float(i_x0);
    i_x0 = atomicCAS(i_ptr, assumed, __float_as_int(@math_kernel@));
  } while (assumed != i_x0);
}

)",
            {
                {"name", name}, {"T", data_type}, {"math_kernel", math_kernel},
            });
    }
    LanguageUnit_p cw(new LanguageUnit(mangled_name, code));
    return cw;
}

uint32_t cuda::align_to_block_size(uint32_t threads, uint32_t block_size)
{
    NNFUSION_CHECK(threads <= (1u << 31) - 1) << "Cuda can't handle threads > 2^31 - 1.";
    uint32_t r = (threads + block_size - 1) / block_size;
    return r;
}

void cuda::emit_memcpyDtD(CodeWriter& writer,
                          shared_ptr<nnfusion::descriptor::Tensor> dst,
                          shared_ptr<nnfusion::descriptor::Tensor> src,
                          size_t buffer_size)
{
    if (buffer_size == 0)
    {
        writer << "CUDA_SAFE_CALL(cudaMemcpyAsync(" << dst->get_name() << ", " << src->get_name()
               << ", " << dst->size() << ", cudaMemcpyDeviceToDevice, stream));\n";
        return;
    }
    writer << "CUDA_SAFE_CALL(cudaMemcpyAsync(" << dst->get_name() << ", " << src->get_name()
           << ", " << buffer_size << ", cudaMemcpyDeviceToDevice, stream));\n";
    return;
}

void cuda::coordinate_transform_to_multi_d(CodeWriter& writer,
                                           std::string i_strides,
                                           std::string i_stride_magic,
                                           std::string i_stride_shift,
                                           std::string i_coord_product,
                                           std::string o_coordinates,
                                           size_t rank,
                                           bool register_arguments)
{
    std::string brace_open = (register_arguments) ? "" : "[";
    std::string brace_close = (register_arguments) ? "" : "]";

    // Translation from flat index to dense tensor coordinates:
    // Given tensor shape [d0 d1 ... dN] with strides [d1*...*dN, d2*...*dN, ... 1],
    // calculate coordinates as:
    //
    //  product = tid
    //  d0 = product/stride[0]
    //  product = product % stride[0]
    //  d1 = product/stride[1]
    //  ...
    writer << "int coordinate_product = " << i_coord_product << ";\n";
    for (size_t i = 0; i < rank; i++)
    {
        writer << "int " << o_coordinates << i << " = division_by_invariant_multiplication("
               << "coordinate_product, " << i_stride_magic << brace_open << i << brace_close << ", "
               << i_stride_shift << brace_open << i << brace_close << ");\n";
        writer << "coordinate_product -= (" << o_coordinates << i << " * " << i_strides
               << brace_open << i << brace_close << ");\n";
    }
}

std::string cuda::collective_coordinate_transform_helper(CodeWriter& writer,
                                                         std::string i_thread_index,
                                                         std::string i_strides,
                                                         std::string i_stride_magic,
                                                         std::string i_stride_shift,
                                                         std::string i_reduced_strides,
                                                         std::string o_coordinates,
                                                         size_t rank,
                                                         bool register_arguments,
                                                         std::string reduced_idx)
{
    coordinate_transform_to_multi_d(writer,
                                    i_strides,
                                    i_stride_magic,
                                    i_stride_shift,
                                    i_thread_index,
                                    o_coordinates,
                                    rank,
                                    register_arguments);

    std::string brace_open = (register_arguments) ? "" : "[";
    std::string brace_close = (register_arguments) ? "" : "]";

    // index into reduced tensor from coordinates of non-reduced tensor
    writer << "uint32_t " << reduced_idx << " = 0;\n";
    for (size_t i = 0; i < rank; i++)
    {
        writer << reduced_idx << " += " << o_coordinates << i << " * " << i_reduced_strides
               << brace_open << i << brace_close << ";\n";
    }
    return reduced_idx;
}

void cuda::div_to_mul(const NVShape& shape, std::vector<int>& magic, std::vector<int>& shift)
{
    for (int i = 0; i < shape.size(); i++)
    {
        int _magic;
        int _shift;
        std::tie(_magic, _shift) = idiv_magic_u64(shape[i]);
        magic.push_back(_magic);
        shift.push_back(_shift);
    }
}

void cuda::simplify_reduce_shape(NVShape in,
                                 NVShape reduce_axis,
                                 NVShape& simplified_shape,
                                 NVShape& simplified_reduce_axis)
{
    int32_t rank = in.size();
    // Sort the axis incase it's not sorted.
    std::sort(reduce_axis.begin(), reduce_axis.end());
    // Clear simplified_shape and axis
    simplified_shape.clear();
    simplified_reduce_axis.clear();
    // Combine axis if there is two or more adjeciant reuce_axis
    // combine axis if there is two or more adjeciant non_reuce_axis
    // update combined shape and axis
    NVShape combined_reduce_axis;
    NVShape adj_map(rank, 0);
    size_t combined_axis_count = 0;
    if (reduce_axis.empty())
    {
        simplified_shape = in;
        simplified_reduce_axis = reduce_axis;
        return;
    }
    for (int32_t i = 0; i < static_cast<int32_t>(reduce_axis[0]) - 1; i++)
    {
        adj_map[i] = 1;
        combined_axis_count++;
    }
    for (int32_t i = 0; i < reduce_axis.size() - 1; i++)
    {
        if (static_cast<int32_t>(reduce_axis[i + 1]) - static_cast<int32_t>(reduce_axis[i]) == 1)
        {
            adj_map[reduce_axis[i]] = 1;
            combined_axis_count++;
        }
        else
        {
            combined_reduce_axis.push_back(reduce_axis[i] - combined_axis_count);
            for (int32_t j = static_cast<int32_t>(reduce_axis[i]) + 1;
                 j < static_cast<int32_t>(reduce_axis[i + 1]) - 1;
                 j++)
            {
                adj_map[j] = 1;
                combined_axis_count++;
            }
        }
    }
    combined_reduce_axis.push_back(reduce_axis.back() - combined_axis_count);
    for (int32_t i = static_cast<int32_t>(reduce_axis.back()) + 1; i < rank - 1; i++)
    {
        adj_map[i] = 1;
    }

    NVShape combined_shape;
    size_t shape_i = 1;
    for (int i = 0; i < rank; i++)
    {
        if (adj_map[i] == 1)
        {
            shape_i *= in[i];
        }
        else
        {
            combined_shape.push_back(shape_i * in[i]);
            shape_i = 1;
        }
    }

    // eleminate dimensons when dimension size = 1, update shape and reduce axis
    size_t reduce_idx = 0;
    size_t eliminated_axis_count = 0;
    for (int32_t i = 0; i < combined_shape.size(); i++)
    {
        if (combined_shape[i] == 1)
        {
            eliminated_axis_count++;
        }
        else
        {
            simplified_shape.push_back(combined_shape[i]);
            if (i == combined_reduce_axis[reduce_idx])
            {
                simplified_reduce_axis.push_back(i - eliminated_axis_count);
            }
        }
        if (reduce_idx < combined_reduce_axis.size() - 1)
        {
            reduce_idx = (i == combined_reduce_axis[reduce_idx]) ? reduce_idx + 1 : reduce_idx;
        }
    }
}

void cuda::get_reduce_strides(NVShape input_shape,
                              NVShape reduce_axis,
                              NVShape& non_reduce_shape,
                              NVShape& non_reduce_strides,
                              NVShape& non_reduce_strides_in_input,
                              NVShape& reduce_shape,
                              NVShape& reduce_strides,
                              NVShape& reduce_strides_in_input)
{
    size_t rank = input_shape.size();
    NVShape reduce_flag(rank, 0);
    for (auto a : reduce_axis)
    {
        reduce_flag[a] = 1;
    }
    NVShape input_strides = nnfusion::row_major_strides(input_shape);
    for (int i = 0; i < rank; i++)
    {
        if (reduce_flag[i] != 0)
        {
            reduce_shape.push_back(input_shape[i]);
            reduce_strides_in_input.push_back(input_strides[i]);
        }
        else
        {
            non_reduce_shape.push_back(input_shape[i]);
            non_reduce_strides_in_input.push_back(input_strides[i]);
        }
    }
    reduce_strides = nnfusion::row_major_strides(reduce_shape);
    non_reduce_strides = nnfusion::row_major_strides(non_reduce_shape);
}