// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"

DECLARE_bool(fextern_result_memory);

#define QUOTE(x) #x
#define STR(x) QUOTE(x)
#define LU_DEFINE(NAME, code)                                                                      \
    LanguageUnit_p NAME = LanguageUnit_p(new LanguageUnit(STR(NAME), code));
#define LU_DEFINE_N(NAME, symbol, code)                                                            \
    LanguageUnit_p NAME = LanguageUnit_p(new LanguageUnit(symbol, code));
using namespace std;

//LanguageUnits
LU_DEFINE_N(header_mpi, "header::mpi", "#include \"mpi.h\"\n");

LU_DEFINE_N(cpu_reference_common, "header::reference_common", R"(
#include "reference_common.h"
using namespace reference_common;
)");

// Two standalone head file

LU_DEFINE(cpu_reference_constant, R"(template <typename T>
void cpu_reference_constant(const T* arg0, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i];
    }
}
)");
LU_DEFINE(cpu_reference_log, R"(template <typename T>
void cpu_reference_log(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::log(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_ceiling, R"(template <typename T>
void cpu_reference_ceiling(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::ceil(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_product, R"(template <typename T>
void cpu_reference_product(const T* arg,
 T* out,
 const Shape in_shape,
 const Shape out_shape,
 const AxisSet reduction_axes)
{
    CoordinateTransform output_transform(out_shape);

    for (const Coordinate output_coord : output_transform)
    {
        out[output_transform.index(output_coord)] = 1;
    }

    CoordinateTransform input_transform(in_shape);

    for (const Coordinate input_coord : input_transform)
    {
        Coordinate output_coord = reduce(input_coord, reduction_axes);

        out[output_transform.index(output_coord)] *=
arg[input_transform.index(input_coord)];
    }
}
)");
LU_DEFINE(cpu_reference_sum, R"(template <typename T>
void cpu_reference_sum(const T* arg,
         T* out,
         const Shape in_shape,
         const Shape out_shape,
         const AxisSet reduction_axes)
{
    CoordinateTransform output_transform(out_shape);

    for (const Coordinate output_coord : output_transform)
    {
        out[output_transform.index(output_coord)] = 0;
    }

    CoordinateTransform input_transform(in_shape);

    for (const Coordinate input_coord : input_transform)
    {
        Coordinate output_coord = reduce(input_coord, reduction_axes);

        out[output_transform.index(output_coord)] +=
arg[input_transform.index(input_coord)];
    }
}
)");
LU_DEFINE(cpu_reference_power, R"(template <typename T>
void cpu_reference_power(const T* arg0, const T* arg1, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::pow(arg0[i], arg1[i]);
    }
}
)");
LU_DEFINE(cpu_reference_cos, R"(template <typename T>
void cpu_reference_cos(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::cpu_reference_cos(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_relu, R"(template <typename T>
void cpu_reference_relu(const T* arg, T* out, size_t count)
{
    T zero = 0;
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg[i] > zero ? arg[i] : zero;
    }
}
template <typename T>
void relu_backprop(const T* arg, T* delta_arg, T* out, size_t count)
{
    T zero = 0;
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg[i] > zero ? delta_arg[i] : zero;
    }
}
)");
LU_DEFINE(cpu_reference_pad, R"(template <typename T>
void cpu_reference_pad(const T* arg0,
         const T* arg1,
         T* out,
         const Shape arg0_shape,
         const Shape out_shape,
         const Shape padding_below,
         const Shape padding_above,
         const Shape padding_interior)
{
    Coordinate input_start(arg0_shape.size(), 0); // start at (0,0,...,0)
    Coordinate input_end =
        out_shape; // end at (d'0,d'1,...,d'n), the outer corner of the post-padding shape

    Strides input_strides(arg0_shape.size(), 1);

    AxisVector input_axis_order(arg0_shape.size());
    for (size_t i = 0; i < arg0_shape.size(); i++)
    {
        input_axis_order[i] = i;
    }

    Strides input_dilation(arg0_shape.size());
    for (size_t i = 0; i < arg0_shape.size(); i++)
    {
        input_dilation[i] = padding_interior[i] + 1;
    }

    // Need to cast these to CoordinateDiff in order to make CoordinateTransform happy.
    CoordinateDiff padding_below_signed;
    CoordinateDiff padding_above_signed;

    for (size_t i = 0; i < padding_below.size(); i++)
    {
        padding_below_signed.push_back(padding_below[i]);
        padding_above_signed.push_back(padding_above[i]);
    }

    CoordinateTransform input_transform(arg0_shape,
    input_start,
    input_end,
    input_strides,
    input_axis_order,
    padding_below_signed,
    padding_above_signed,
    input_dilation);
    CoordinateTransform output_transform(out_shape);

    CoordinateTransform::Iterator output_it = output_transform.begin();

    assert(shape_size(input_transform.get_target_shape()) ==
      shape_size(output_transform.get_target_shape()));

    for (const Coordinate in_coord : input_transform)
    {
        const Coordinate out_coord = *output_it;

        T v = input_transform.has_source_coordinate(in_coord)
      ? arg0[input_transform.index(in_coord)]
      : *arg1;

        out[output_transform.index(out_coord)] = v;

        ++output_it;
    }
}
)");
LU_DEFINE(cpu_reference_batch_norm, R"(template <typename T>
void cpu_reference_batch_norm(double eps,
   const T* arg0,
   const T* arg1,
   const T* arg2,
   const T* arg3,
   const T* arg4,
   T* out0,
   const Shape arg2_shape)
{
    auto eps_casted = static_cast<T>(eps);
    CoordinateTransform arg2_transform(arg2_shape);

    for (Coordinate arg2_coord : arg2_transform)
    {
        auto channel_num = arg2_coord[1];
        auto channel_gamma = arg0[channel_num];
        auto channel_beta = arg1[channel_num];
        auto channel_mean = arg3[channel_num];
        auto channel_var = arg4[channel_num];

        auto input_index = arg2_transform.index(arg2_coord);
        auto normalized =
(arg2[input_index] - channel_mean) / (std::sqrt(channel_var + eps_casted));
        out0[input_index] = normalized * channel_gamma + channel_beta;
    }
}
)");
LU_DEFINE(cpu_reference_cosh, R"(template <typename T>
void cpu_reference_cosh(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::cpu_reference_cosh(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_lrn, R"(template <typename T>
void cpu_reference_lrn(const T* arg,
         T* out,
         const Shape arg_shape,
         double dalpha,
         double dbeta,
         double dbias,
         size_t size)
{
    T alpha = static_cast<T>(dalpha);
    T beta = static_cast<T>(dbeta);
    T bias = static_cast<T>(dbias);

    CoordinateTransform input_transform(arg_shape);
    const size_t CHANNEL_DIM = 1;
    const size_t MAX_C = arg_shape.at(CHANNEL_DIM);
    for (const Coordinate in_coord : input_transform)
    {
        size_t c = in_coord.at(CHANNEL_DIM);
        T square_sum = 0;
        for (size_t i = c; i < c + size; i++)
        {
if (i < (size - 1) / 2)
    continue;
if (i >= MAX_C + (size - 1) / 2)
    continue;
auto sum_coord = in_coord;
sum_coord.at(CHANNEL_DIM) = i - (size - 1) / 2;
square_sum += arg[input_transform.index(sum_coord)] *
  arg[input_transform.index(sum_coord)];
        }

        T x = arg[input_transform.index(in_coord)];
        out[input_transform.index(in_coord)] =
x / (std::pow(bias + (alpha / size) * square_sum, beta));
    }
}
)");
LU_DEFINE(cpu_reference_result, R"(template <typename T>
void cpu_reference_result(const T* arg, T* out, size_t count)
{
    memcpy(out, arg, sizeof(T) * count);
}
)");
LU_DEFINE(cpu_reference_concat, R"(template <typename T>
void cpu_reference_concat(const std::vector<const T*>& args,
T* out,
const std::vector<Shape>& in_shapes,
const Shape out_shape,
size_t concatenation_axis)
{
    // We will copy the inputs to the output one at a time. As we go, we will move out along the
    // concatenation axis, starting at 0.
    size_t concatenation_pos = 0;

    for (size_t i = 0; i < args.size(); i++)
    {
        // CoordinateTransform gets confused when the last input has a zero-size dim, so we will
        // just skip for zero-element tensors.
        if (shape_size(in_shapes[i]) == 0)
        {
continue;
        }

        // The start coordinate for the copy is (0,...,0) except at the concatenation axis.
        Coordinate out_start_coord(out_shape.size(), 0);
        out_start_coord[concatenation_axis] = concatenation_pos;

        // The end coordinate for the copy is the same as the output shape except at the
        // concatenation axis.
        Coordinate out_end_coord = out_shape;
        out_end_coord[concatenation_axis] =
concatenation_pos + in_shapes[i][concatenation_axis];

        CoordinateTransform input_transform(in_shapes[i]);
        CoordinateTransform output_chunk_transform(
out_shape, out_start_coord, out_end_coord);

        assert(shape_size(input_transform.get_target_shape()) ==
          shape_size(output_chunk_transform.get_target_shape()));

        CoordinateTransform::Iterator output_chunk_it = output_chunk_transform.begin();

        for (const Coordinate input_coord : input_transform)
        {
size_t input_index = input_transform.index(input_coord);
size_t output_chunk_index = output_chunk_transform.index(*output_chunk_it);
++output_chunk_it;

out[output_chunk_index] = args[i][input_index];
        }

        concatenation_pos += in_shapes[i][concatenation_axis];
    }
}
)");
LU_DEFINE(cpu_reference_greater, R"(template <typename T>
void cpu_reference_greater(const T* arg0,
 const T* arg1,
 char* out,
 size_t count) // TODO: using char for bool, is this right?
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i] > arg1[i];
    }
}
)");
LU_DEFINE(cpu_reference_max, R"(template <typename T>
void cpu_reference_max(const T* arg,
         T* out,
         const Shape in_shape,
         const Shape out_shape,
         const AxisSet reduction_axes)
{
    T minval = std::numeric_limits<T>::has_infinity
       ? -std::numeric_limits<T>::infinity()
       : std::numeric_limits<T>::min();

    CoordinateTransform output_transform(out_shape);

    for (const Coordinate output_coord : output_transform)
    {
        out[output_transform.index(output_coord)] = minval;
    }

    CoordinateTransform input_transform(in_shape);

    for (const Coordinate input_coord : input_transform)
    {
        Coordinate output_coord = reduce(input_coord, reduction_axes);

        T x = arg[input_transform.index(input_coord)];
        T max = out[output_transform.index(output_coord)];
        if (x > max)
        {
out[output_transform.index(output_coord)] = x;
        }
    }
}
)");
LU_DEFINE(cpu_reference_broadcast, R"(template <typename T>
void cpu_reference_broadcast(const T* arg,
   T* out,
   const Shape in_shape,
   const Shape out_shape,
   const AxisSet broadcast_axes)
{
    CoordinateTransform input_transform(in_shape);
    CoordinateTransform output_transform(out_shape);

    for (const Coordinate output_coord : output_transform)
    {
        Coordinate input_coord = reduce(output_coord, broadcast_axes);

        out[output_transform.index(output_coord)] =
arg[input_transform.index(input_coord)];
    }
}
)");
LU_DEFINE(cpu_reference_or, R"(template <typename T>
void logical_cpu_reference_or(const T* arg0, const T* arg1, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = static_cast<T>(arg0[i] || arg1[i]);
    }
}
)");
LU_DEFINE(cpu_reference_equal, R"(template <typename T>
void cpu_reference_equal(const T* arg0,
           const T* arg1,
           char* out,
           size_t count) // TODO: using char for bool, is this right?
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i] == arg1[i];
    }
}
)");
LU_DEFINE(cpu_reference_convolution, R"(template <typename T>
void cpu_reference_convolution(const T* arg0,
     const T* arg1,
     T* out,
     const Shape arg0_shape,
     const Shape arg1_shape,
     const Shape out_shape,
     const Strides window_movement_strides,
     const Strides window_dilation_strides,
     const CoordinateDiff padding_below,
     const CoordinateDiff padding_above,
     const Strides data_dilation_strides,
     size_t batch_axis_data,
     size_t input_channel_axis_data,
     size_t input_channel_axis_filters,
     size_t output_channel_axis_filters,
     size_t batch_axis_result,
     size_t output_channel_axis_result,
     bool rotate_filter)
{
    // Comments throughout assume without loss of generality that:
    //
    // * batch axes for both input data and output data are 0
    // * input channel axes for both input data and filters are 1
    // * output channel axes for filters is 0
    // * output channel axis for output data is 1
    // * rotate_filter is false

    // At the outermost level we will walk over every output coordinate O.
    CoordinateTransform output_transform(out_shape);

    for (Coordinate out_coord : output_transform)
    {
        // Our output coordinate O will have the form:
        //
        //   (N,chan_out,i_1,...,i_n)

        size_t batch_index = out_coord[batch_axis_result];
        size_t output_channel = out_coord[output_channel_axis_result];

        // For the input data we need to iterate the coordinate:
        //
        //   I:
        //
        // over the range (noninclusive on the right):
        //
        //   (N,0,s_1*i_1,s_2*i_2,...,s_n*i_n) ->
        //
        //     (N+1,chans_in_count,s_1*i_1 + l_1*filter_dims_1,...,s_n*i_n + l_n*filter_dims_n)
        //
        // with strides:
        //
        //   (1,l_1,...,l_n).
        //
        // Note that we are iterating within the *padded* and *dilated* data batch, so further
        // down we must check the current coordinate is in the padding or dilation gap.

        size_t n_spatial_dimensions = arg0_shape.size() - 2;
        size_t n_input_channels = arg0_shape[input_channel_axis_data];

        Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
        Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
        Strides input_batch_transform_movement_strides(2 + n_spatial_dimensions, 1);
        CoordinateDiff input_batch_transform_padding_below(2 + n_spatial_dimensions, 0);
        CoordinateDiff input_batch_transform_padding_above(2 + n_spatial_dimensions, 0);
        Strides input_batch_transform_dilation_strides(2 + n_spatial_dimensions, 1);

        input_batch_transform_start[batch_axis_data] = batch_index;
        input_batch_transform_end[batch_axis_data] = batch_index + 1;
        input_batch_transform_start[input_channel_axis_data] = 0;
        input_batch_transform_end[input_channel_axis_data] = n_input_channels;

        for (size_t i = 2; i < n_spatial_dimensions + 2; i++)
        {
size_t window_dilation_stride = window_dilation_strides[i - 2];
size_t window_movement_stride = window_movement_strides[i - 2];
std::ptrdiff_t below_pad = padding_below[i - 2];
std::ptrdiff_t above_pad = padding_above[i - 2];
size_t data_dilation_stride = data_dilation_strides[i - 2];

input_batch_transform_start[i] = window_movement_stride * out_coord[i];
input_batch_transform_end[i] =
    input_batch_transform_start[i] +
    (arg1_shape[i] - 1) * window_dilation_stride + 1;
input_batch_transform_movement_strides[i] = window_dilation_stride;
input_batch_transform_padding_below[i] = below_pad;
input_batch_transform_padding_above[i] = above_pad;
input_batch_transform_dilation_strides[i] = data_dilation_stride;
        }

        AxisVector input_batch_transform_axis_order(2 + n_spatial_dimensions);
        for (size_t i = 0; i < input_batch_transform_axis_order.size(); i++)
        {
input_batch_transform_axis_order[i] = i;
        }

        CoordinateTransform input_batch_transform(
arg0_shape,
input_batch_transform_start,
input_batch_transform_end,
input_batch_transform_movement_strides,
input_batch_transform_axis_order,
input_batch_transform_padding_below,
input_batch_transform_padding_above,
input_batch_transform_dilation_strides);

        // Simultaneously with iterating I, for the filters we need to iterate the coordinate:
        //
        //   F
        //
        // over the range (noninclusive on the right):
        //
        //   (chan_out,0,0,...,0) -> (chan_out+1,chans_in_count,filter_dims_1,...,filter_dims_n)
        //
        // with unit stride.

        Shape filter_transform_start(2 + n_spatial_dimensions);
        Shape filter_transform_end(2 + n_spatial_dimensions);

        filter_transform_start[output_channel_axis_filters] = output_channel;
        filter_transform_end[output_channel_axis_filters] = output_channel + 1;
        filter_transform_start[input_channel_axis_filters] = 0;
        filter_transform_end[input_channel_axis_filters] = n_input_channels;

        for (size_t i = 2; i < n_spatial_dimensions + 2; i++)
        {
filter_transform_start[i] = 0;
filter_transform_end[i] = arg1_shape[i];
        }

        CoordinateTransform filter_transform(
arg1_shape, filter_transform_start, filter_transform_end);

        // As we go, we sum up:
        //
        //   output[O] += arg0[I] * arg1[F].

        T result = 0;

        CoordinateTransform::Iterator input_it = input_batch_transform.begin();
        CoordinateTransform::Iterator filter_it = filter_transform.begin();

        while (input_it != input_batch_transform.end() &&
   filter_it != filter_transform.end())
        {
const Coordinate input_batch_coord = *input_it;
Coordinate filter_coord = *filter_it;

if (rotate_filter)
{
    Shape target_shape = filter_transform.get_target_shape();

    // Note that we only reverse the spatial dimensions here (loop
    // starts at 2)
    for (size_t i = 2; i < filter_coord.size(); i++)
    {
        filter_coord[i] = target_shape[i] - filter_coord[i] - 1;
    }
}

T v = input_batch_transform.has_source_coordinate(input_batch_coord)
          ? arg0[input_batch_transform.index(input_batch_coord)]
          : 0;

result += v * arg1[filter_transform.index(filter_coord)];

++input_it;
++filter_it;
        }

        out[output_transform.index(out_coord)] = result;
    }
}
)");
LU_DEFINE(cpu_reference_dequantize, R"(template <typename QUANT, typename REAL>
void cpu_reference_dequantize(const QUANT* input,
    const REAL* scale,
    const QUANT* offset,
    REAL* output,
    const Shape input_shape,
    const Shape scale_offset_shape,
    const AxisSet axes)
{
    CoordinateTransform input_transform(input_shape);
    CoordinateTransform scale_offset_transform(scale_offset_shape);

    for (const Coordinate input_coord : input_transform)
    {
        Coordinate scale_offset_coord = project(input_coord, axes);

        output[input_transform.index(input_coord)] =
static_cast<REAL>(
    (input[input_transform.index(input_coord)] -
     offset[scale_offset_transform.index(scale_offset_coord)])) *
scale[scale_offset_transform.index(scale_offset_coord)];
    }
}
)");
LU_DEFINE(cpu_reference_select_and_scatter, R"(template <typename T>
void cpu_reference_select_and_scatter(const T* arg_selectee,
const T* arg_source,
const T* arg_init,
T* out,
const Shape arg_selectee_shape,
const Shape arg_source_shape,
const Shape out_shape,
std::function<char(T, T)> selection_function,
std::function<T(T, T)> scatter_function,
const Shape window_shape,
const Strides window_movement_strides)
{
    // First write every element of the output with the supplied initial value.
    CoordinateTransform output_transform(out_shape);

    for (const Coordinate out_coord : output_transform)
    {
        out[output_transform.index(out_coord)] = *arg_init;
    }

    // Slide the window over selectee/output.
    Shape window_start_corner_transform_start(arg_selectee_shape.size(), 0);
    Shape window_start_corner_transform_end(arg_selectee_shape.size());

    for (size_t i = 0; i < arg_selectee_shape.size(); i++)
    {
        window_start_corner_transform_end[i] =
arg_selectee_shape[i] - window_shape[i] + 1;
    }

    CoordinateTransform window_start_corner_transform(
        arg_selectee_shape,
        window_start_corner_transform_start,
        window_start_corner_transform_end,
        window_movement_strides);

    CoordinateTransform source_transform(arg_source_shape);
    CoordinateTransform::Iterator source_it = source_transform.begin();

    for (Coordinate window_start_coord : window_start_corner_transform)
    {
        // We need a physical rather than virtual coordinate to start the window.
        window_start_coord =
window_start_corner_transform.to_source_coordinate(window_start_coord);

        Shape window_transform_end(arg_selectee_shape.size());
        for (size_t i = 0; i < arg_selectee_shape.size(); i++)
        {
window_transform_end[i] = window_start_coord[i] + window_shape[i];
        }

        CoordinateTransform window_transform(
arg_selectee_shape, window_start_coord, window_transform_end);

        bool first_val = true;
        Coordinate winner_coord;

        // This initial value is ignored; it's just here so the compiler knows
        // for sure that winner_val is initialized.
        T winner_val = 0;

        for (const Coordinate challenger_coord : window_transform)
        {
T challenger_val = arg_selectee[window_transform.index(challenger_coord)];

if (first_val || selection_function(challenger_val, winner_val))
{
    winner_coord = challenger_coord;
    winner_val = challenger_val;
    first_val = false;
}
        }

        Coordinate source_coord = *source_it;

        T old_output_val = out[window_transform.index(winner_coord)];
        T source_val = arg_source[source_transform.index(source_coord)];
        T new_output_val = scatter_function(old_output_val, source_val);

        out[window_transform.index(winner_coord)] = new_output_val;

        ++source_it;
    }
}
)");
LU_DEFINE(cpu_reference_convert, R"(template <typename TI, typename TO>
void cpu_reference_convert(const TI* arg, TO* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = static_cast<TO>(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_quantize, R"(template <typename REAL, typename QUANT>
void cpu_reference_quantize(const REAL* input,
  const REAL* scale,
  const QUANT* offset,
  QUANT* output,
  const Shape input_shape,
  const Shape scale_offset_shape,
  const AxisSet axes,
  op::Quantize::RoundMode round_mode)
{
    CoordinateTransform input_transform(input_shape);
    CoordinateTransform scale_offset_transform(scale_offset_shape);

    for (const Coordinate input_coord : input_transform)
    {
        Coordinate scale_offset_coord = project(input_coord, axes);

        // apply scale
        REAL qvalue = input[input_transform.index(input_coord)] /
          scale[scale_offset_transform.index(scale_offset_coord)];

        // round
        if (round_mode == op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY ||
round_mode == op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO)
        {
auto abs_qvalue = std::fabs(qvalue);
auto abs_qvalue_toward_inf = std::floor(abs_qvalue + 0.5);
qvalue = (qvalue < 0.0) ? -abs_qvalue_toward_inf : abs_qvalue_toward_inf;
        }
        else if (round_mode == op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_ZERO)
        {
auto abs_qvalue = std::fabs(qvalue);
auto abs_qvalue_toward_zero = std::ceil(abs_qvalue - 0.5);
qvalue = (qvalue < 0.0) ? -abs_qvalue_toward_zero : abs_qvalue_toward_zero;
        }
        else if (round_mode == op::Quantize::RoundMode::ROUND_NEAREST_UPWARD)
        {
qvalue = std::floor(qvalue + 0.5);
        }
        else if (round_mode == op::Quantize::RoundMode::ROUND_NEAREST_DOWNWARD)
        {
qvalue = std::ceil(qvalue - 0.5);
        }
        else if (round_mode == op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN)
        {
auto up_qvalue = std::floor(qvalue + 0.5);
auto dn_qvalue = std::ceil(qvalue - 0.5);
auto rem = std::fmod(up_qvalue, 2.0);
qvalue = (rem == 0.0) ? up_qvalue : dn_qvalue;
        }
        else if (round_mode == op::Quantize::RoundMode::ROUND_TOWARD_INFINITY)
        {
auto abs_qvalue = std::fabs(qvalue);
auto abs_qvalue_toward_inf = std::ceil(abs_qvalue);
qvalue = (qvalue < 0.0) ? -abs_qvalue_toward_inf : abs_qvalue_toward_inf;
        }
        else if (round_mode == op::Quantize::RoundMode::ROUND_TOWARD_ZERO)
        {
auto abs_qvalue = std::fabs(qvalue);
auto abs_qvalue_toward_zero = std::floor(abs_qvalue);
qvalue = (qvalue < 0.0) ? -abs_qvalue_toward_zero : abs_qvalue_toward_zero;
        }
        else if (round_mode == op::Quantize::RoundMode::ROUND_UP)
        {
qvalue = std::ceil(qvalue);
        }
        else if (round_mode == op::Quantize::RoundMode::ROUND_DOWN)
        {
qvalue = std::floor(qvalue);
        }

        // apply offset
        qvalue += offset[scale_offset_transform.index(scale_offset_coord)];

        // clamp
        qvalue = std::max<REAL>(qvalue,
        static_cast<REAL>(std::numeric_limits<QUANT>::min()));
        qvalue = std::min<REAL>(qvalue,
        static_cast<REAL>(std::numeric_limits<QUANT>::max()));

        // cast
        output[input_transform.index(input_coord)] = static_cast<QUANT>(qvalue);
    }
}
)");

// <todo:wenxh> allreduce need MPI
LU_DEFINE(cpu_reference_allreduce,
          R"(void cpu_reference_allreduce_float(const float* arg, float* out, int count)
{
    MPI_Allreduce(arg, out, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}
)");
LU_DEFINE(cpu_reference_argmax, R"(template <typename T, typename U>
void cpu_reference_argmax(
    const T* arg, U* out, const Shape in_shape, const Shape out_shape, size_t axis)
{
    // take the first elements (i.e. 0 indices) in out_shape - axis as maximums
    memset(out, 0, shape_size(out_shape) * sizeof(U));

    AxisVector av{axis};
    CoordinateTransform input_transform(in_shape);

    for (const Coordinate input_coord : input_transform)
    {
        Coordinate output_coord = reduce(input_coord, av);
        CoordinateTransform output_transform(out_shape);

        auto min_index = static_cast<size_t>(out[output_transform.index(output_coord)]);
        auto min_coord = input_coord;
        min_coord[axis] = min_index;
        if (arg[input_transform.index(input_coord)] >
arg[input_transform.index(min_coord)])
        {
out[output_transform.index(output_coord)] =
    static_cast<U>(input_coord[axis]);
        }
    }
}
)");
LU_DEFINE(cpu_reference_and, R"(template <typename T>
void logical_cpu_reference_and(const T* arg0, const T* arg1, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = static_cast<T>(arg0[i] && arg1[i]);
    }
}
)");
LU_DEFINE(cpu_reference_argmin, R"(template <typename T, typename U>
void cpu_reference_argmin(
    const T* arg, U* out, const Shape in_shape, const Shape out_shape, size_t axis)
{
    // take the first elements (i.e. 0 indices) in out_shape - axis as minimums
    memset(out, 0, shape_size(out_shape) * sizeof(U));

    AxisVector av{axis};
    CoordinateTransform input_transform(in_shape);

    for (const Coordinate input_coord : input_transform)
    {
        Coordinate output_coord = reduce(input_coord, av);
        CoordinateTransform output_transform(out_shape);

        auto min_index = static_cast<size_t>(out[output_transform.index(output_coord)]);
        auto min_coord = input_coord;
        min_coord[axis] = min_index;
        if (arg[input_transform.index(input_coord)] <
arg[input_transform.index(min_coord)])
        {
out[output_transform.index(output_coord)] =
    static_cast<U>(input_coord[axis]);
        }
    }
}
)");
LU_DEFINE(cpu_reference_copy, R"(template <typename T>
void cpu_reference_copy(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg[i];
    }
}
)");
LU_DEFINE(cpu_reference_reduce, R"(template <typename T>
void cpu_reference_reduce(const T* arg0,
const T* arg1, // TODO: really we should just pass a T here.
T* out,
const Shape in_shape,
const Shape out_shape,
const AxisSet reduction_axes,
std::function<T(T, T)> reduction_function)
{
    CoordinateTransform output_transform(out_shape);

    for (const Coordinate output_coord : output_transform)
    {
        out[output_transform.index(output_coord)] = *arg1;
    }

    CoordinateTransform input_transform(in_shape);

    for (const Coordinate input_coord : input_transform)
    {
        Coordinate output_coord = reduce(input_coord, reduction_axes);
        size_t input_index = input_transform.index(input_coord);
        size_t output_index = output_transform.index(output_coord);

        out[output_index] = reduction_function(out[output_index], arg0[input_index]);
    }
}
)");
LU_DEFINE(cpu_reference_less_eq, R"(template <typename T>
void cpu_reference_less_eq(const T* arg0,
 const T* arg1,
 char* out,
 size_t count) // TODO: using char for bool, is this right?
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i] <= arg1[i];
    }
}
)");
LU_DEFINE(cpu_reference_multiply, R"(template <typename T>
void cpu_reference_multiply(const T* arg0, const T* arg1, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i] * arg1[i];
    }
}
)");
LU_DEFINE(cpu_reference_sigmoid, R"(template <typename T>
void cpu_reference_sigmoid(const T* arg, T* out, size_t count)
{
    T exp_value;
    for (size_t i = 0; i < count; i++)
    {
        exp_value = std::exp(-arg[i]);
        out[i] = 1 / (1 + exp_value);
    }
}

template <typename T>
void sigmoid_backprop(const T* arg, T* delta_arg, T* out, size_t count)
{
    T exp_value;
    T func_x;
    for (size_t i = 0; i < count; i++)
    {
        exp_value = std::exp(-arg[i]);
        func_x = 1 / (1 + exp_value);
        out[i] = delta_arg[i] * func_x * (1 - func_x);
    }
}
)");
LU_DEFINE(cpu_reference_sign, R"(template <typename T>
void cpu_reference_sign(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = (arg[i] < 0 ? -1 : (arg[i] > 0 ? 1 : 0));
    }
}
)");
LU_DEFINE(cpu_reference_topk, R"(template <typename T, typename U>
void cpu_reference_topk(const T* arg,
          U* out_indices,
          T* out_values,
          const Shape in_shape,
          const Shape out_shape,
          size_t axis,
          size_t k,
          bool compute_max)
{
    using namespace std;
    // reorder source axis visit order and make "axis" inner most
    size_t ndim = static_cast<size_t>(in_shape.size());
    Coordinate start_corner(ndim, 0);
    Coordinate end_corner(in_shape);
    end_corner[axis] = 1;
    Strides strides(ndim, 1);
    AxisVector axis_order(ndim);
    iota(axis_order.begin(), axis_order.end(), 0);
    axis_order.erase(axis_order.begin() + axis);
    axis_order.push_back(axis);
    // Create CoordinateTransforms that visits only the first element along "axis"
    CoordinateTransform input_transform(
        in_shape, start_corner, end_corner, strides, axis_order);
    CoordinateTransform output_transform(
        out_shape, start_corner, end_corner, strides, axis_order);
    // Create temp vector for sorting.
    vector<tuple<T, U>> workspace(in_shape[axis]);
    vector<size_t> in_strides = nnfusion::row_major_strides(in_shape);
    vector<size_t> out_strides = nnfusion::row_major_strides(out_shape);
    auto in_axis_stride = in_strides[axis];
    auto out_axis_stride = out_strides[axis];
    for (const Coordinate coord : input_transform)
    {
        auto arg_index = input_transform.index(coord);
        auto out_index = output_transform.index(coord);
        // Fill the temp vector
        U i = 0;
        for (tuple<T, U>& entry : workspace)
        {
get<0>(entry) = arg[arg_index];
get<1>(entry) = i;
arg_index += in_axis_stride;
i++;
        }
        // Sort the temp vector
        sort(
workspace.begin(),
workspace.end(),
compute_max
? [](const tuple<T, U>& a, const tuple<T, U>& b) -> bool { return a > b; }
: [](const tuple<T, U>& a, const tuple<T, U>& b) -> bool { return a < b; });
        // Write temp vector to output
        for (size_t j = 0; j < k; j++)
        {
tuple<T, U> entry = workspace[j];
out_values[out_index] = get<0>(entry);
out_indices[out_index] = get<1>(entry);
out_index += out_axis_stride;
        }
    }
}
)");
LU_DEFINE(cpu_reference_asin, R"(template <typename T>
void cpu_reference_asin(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::cpu_reference_asin(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_min, R"(template <typename T>
void cpu_reference_min(const T* arg,
         T* out,
         const Shape in_shape,
         const Shape out_shape,
         const AxisSet reduction_axes)
{
    T minval = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity()
    : std::numeric_limits<T>::max();

    CoordinateTransform output_transform(out_shape);

    for (const Coordinate output_coord : output_transform)
    {
        out[output_transform.index(output_coord)] = minval;
    }

    CoordinateTransform input_transform(in_shape);

    for (const Coordinate input_coord : input_transform)
    {
        Coordinate output_coord = reduce(input_coord, reduction_axes);

        T x = arg[input_transform.index(input_coord)];
        T min = out[output_transform.index(output_coord)];
        if (x < min)
        {
out[output_transform.index(output_coord)] = x;
        }
    }
}
)");
LU_DEFINE(cpu_reference_reduce_window, R"(template <typename T>
void cpu_reference_reduce_window(const T* arg_reductee,
       const T* arg_init,
       T* out,
       const Shape arg_reductee_shape,
       const Shape out_shape,
       std::function<T(T, T)> reduction_function,
       const Shape window_shape,
       const Strides window_movement_strides)
{
    // At the outermost level we will walk over every output coordinate O.
    CoordinateTransform output_transform(out_shape);

    for (const Coordinate out_coord : output_transform)
    {
        // Our output coordinate O will have the form:
        //
        //   (i_1,...,i_n)
        //
        // For the reductee we need to iterate the coordinate:
        //
        //   I:
        //
        // over the range (noninclusive on the right):
        //
        //   (s_1*i_1,s_2*i_2,...,s_n*i_n) ->
        //
        //     (s_1*i_1 + window_shape_1,...,s_n*i_n + window_shape_n)
        //
        // with unit stride.

        Shape reductee_transform_start;
        Shape reductee_transform_end;

        for (size_t i = 0; i < arg_reductee_shape.size(); i++)
        {
size_t window_shape_this_dim = window_shape[i];
size_t movement_stride = window_movement_strides[i];

reductee_transform_start.push_back(movement_stride * out_coord[i]);
reductee_transform_end.push_back(reductee_transform_start[i] +
         window_shape_this_dim);
        }

        CoordinateTransform reductee_transform(
arg_reductee_shape, reductee_transform_start, reductee_transform_end);

        // As we go, we compute the reduced value:
        //
        //   output[O] := reduction_function(output[O],arg[I])

        T result = *arg_init;

        for (const Coordinate reductee_coord : reductee_transform)
        {
result = reduction_function(
    result, arg_reductee[reductee_transform.index(reductee_coord)]);
        }

        out[output_transform.index(out_coord)] = result;
    }
}
)");
LU_DEFINE(cpu_reference_less, R"(template <typename T>
void cpu_reference_less(const T* arg0,
          const T* arg1,
          char* out,
          size_t count) // TODO: using char for bool, is this right?
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i] < arg1[i];
    }
}
)");
LU_DEFINE(cpu_reference_add, R"(template <typename T>
void cpu_reference_add(const T* arg0, const T* arg1, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i] + arg1[i];
    }
}
)");
LU_DEFINE(cpu_reference_floor, R"(template <typename T>
void cpu_reference_floor(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::floor(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_reshape, R"(template <typename T>
void cpu_reference_reshape(const T* arg,
 T* out,
 const Shape in_shape,
 const AxisVector in_axis_order,
 const Shape out_shape)
{
    // Unfortunately we don't yet have a constructor for CoordinateTransform that lets us pass only source_space_shape
    // and source_axis_order so we have to construct the defaults here.
    Shape in_start_corner(in_shape.size(), 0); // (0,...0)
    Strides in_strides(in_shape.size(), 1);    // (1,...,1)

    CoordinateTransform input_transform(
        in_shape, in_start_corner, in_shape, in_strides, in_axis_order);
    CoordinateTransform output_transform(out_shape);

    assert(shape_size(input_transform.get_target_shape()) ==
      shape_size(output_transform.get_target_shape()));

    CoordinateTransform::Iterator output_it = output_transform.begin();

    for (const Coordinate input_coord : input_transform)
    {
        const Coordinate output_coord = *output_it;

        out[output_transform.index(output_coord)] =
arg[input_transform.index(input_coord)];

        ++output_it;
    }
}
)");
LU_DEFINE(cpu_reference_tan, R"(template <typename T>
void cpu_reference_tan(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::cpu_reference_tan(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_abs, R"(template <typename T>
void cpu_reference_abs(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        // TODO: generic "abs" doesn't work here for some reason.
        out[i] = (arg[i] < 0 ? -arg[i] : arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_not, R"(template <typename T>
void logical_cpu_reference_not(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = static_cast<T>(!(arg[i]));
    }
}
)");
LU_DEFINE(cpu_reference_sinh, R"(template <typename T>
void cpu_reference_sinh(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::cpu_reference_sinh(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_not_equal, R"(template <typename T>
void cpu_reference_not_equal(const T* arg0,
   const T* arg1,
   char* out,
   size_t count) // TODO: using char for bool, is this right?
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i] != arg1[i];
    }
}
)");
LU_DEFINE(cpu_reference_reverse_sequence, R"(template <typename T, typename U>
void cpu_reference_reverse_sequence(const T* arg,
          T* out,
          const Shape arg_shape,
          size_t batch_axis,
          size_t sequence_axis,
          U* sequence_lengths)
{
    CoordinateTransform input_transform(arg_shape);
    for (const Coordinate in_coord : input_transform)
    {
        size_t batch_index = in_coord[batch_axis];
        auto orig_seq_index = static_cast<size_t>(sequence_lengths[batch_index]);

        if (orig_seq_index > arg_shape.at(sequence_axis))
        {
throw ngraph_error(
    "One of the elements of sequence lengths is greater than sequence axis "
    "dimension");
        }

        if (orig_seq_index == 0)
        {
orig_seq_index = 1;
        }

        size_t sequence_index = in_coord[sequence_axis] < orig_seq_index
? orig_seq_index - in_coord[sequence_axis] - 1
: in_coord[sequence_axis];

        // make a copy of in_coord and update sequence_index
        Coordinate out_coord = in_coord;
        out_coord[sequence_axis] = sequence_index;
        out[input_transform.index(out_coord)] = arg[input_transform.index(in_coord)];
    }
}
)");
LU_DEFINE(cpu_reference_max_pool, R"(template <typename T>
void max_pool_backprop(const T* arg_forward,
           const T* delta,
           T* out,
           const Shape delta_shape,
           const Shape out_shape, // same as arg_forward_shape
           const Shape window_shape,
           const Strides window_movement_strides,
           const Shape padding_below,
           const Shape padding_above)
{
    CoordinateTransform out_transform(out_shape);

    for (const Coordinate out_coord : out_transform)
    {
        out[out_transform.index(out_coord)] = 0;
    }

    CoordinateTransform delta_transform(delta_shape);

    for (const Coordinate delta_coord : delta_transform)
    {
        size_t img_index = delta_coord[0];
        size_t channel = delta_coord[1];

        size_t n_image_dimensions = out_shape.size() - 2;
        Coordinate source_window_transform_start(2 + n_image_dimensions);
        Coordinate source_window_transform_end(2 + n_image_dimensions);
        Strides source_window_transform_source_strides(2 + n_image_dimensions, 1);
        AxisVector source_window_transform_source_axis_order(2 + n_image_dimensions);
        CoordinateDiff source_window_transform_padding_below(2 + n_image_dimensions);
        CoordinateDiff source_window_transform_padding_above(2 + n_image_dimensions);

        source_window_transform_start[0] = img_index;
        source_window_transform_end[0] = img_index + 1;
        source_window_transform_start[1] = channel;
        source_window_transform_end[1] = channel + 1;
        source_window_transform_padding_below[0] = 0;
        source_window_transform_padding_below[1] = 0;
        source_window_transform_padding_above[0] = 0;
        source_window_transform_padding_above[1] = 0;

        for (size_t i = 2; i < n_image_dimensions + 2; i++)
        {
size_t window_shape_this_dim = window_shape[i - 2];
size_t movement_stride = window_movement_strides[i - 2];

source_window_transform_start[i] = movement_stride * delta_coord[i];
source_window_transform_end[i] =
    source_window_transform_start[i] + window_shape_this_dim;
source_window_transform_padding_below[i] = padding_below[i - 2];
source_window_transform_padding_above[i] = padding_above[i - 2];
        }
        std::iota(begin(source_window_transform_source_axis_order),
      end(source_window_transform_source_axis_order),
      0);

        CoordinateTransform source_window_transform(
out_shape,
source_window_transform_start,
source_window_transform_end,
source_window_transform_source_strides,
source_window_transform_source_axis_order,
source_window_transform_padding_below,
source_window_transform_padding_above);

        Coordinate argmax_coord;
        bool argmax_coord_valid = false;
        T max_val = 0; // just initializing to keep compiler happy, this 0 is ignored

        for (const Coordinate source_window_coord : source_window_transform)
        {
if (source_window_transform.has_source_coordinate(source_window_coord))
{
    T candidate =
        arg_forward[source_window_transform.index(source_window_coord)];

    if (!argmax_coord_valid || candidate > max_val)
    {
        max_val = candidate;
        argmax_coord = source_window_coord;
        argmax_coord_valid = true;
    }
}
        }

        if (argmax_coord_valid)
        {
out[source_window_transform.index(argmax_coord)] +=
    delta[delta_transform.index(delta_coord)];
        }
    }
}

template <typename T>
void cpu_reference_max_pool(const T* arg,
  T* out,
  const Shape arg_shape,
  const Shape out_shape,
  const Shape window_shape,
  const Strides window_movement_strides,
  const Shape padding_below,
  const Shape padding_above)
{
    // At the outermost level we will walk over every output coordinate O.
    CoordinateTransform output_transform(out_shape);

    for (const Coordinate out_coord : output_transform)
    {
        // Our output coordinate O will have the form:
        //
        //   (N,chan,i_1,...,i_n)

        size_t batch_index = out_coord[0];
        size_t channel = out_coord[1];

        // For the input data we need to iterate the coordinate:
        //
        //   I:
        //
        // over the range (noninclusive on the right):
        //
        //   (N,chan,s_1*i_1,s_2*i_2,...,s_n*i_n) ->
        //
        //     (N+1,chan+1,s_1*i_1 + window_shape_1,...,s_n*i_n + window_shape_n)
        //
        // with unit stride.
        //
        // We iterate this over the *padded* data, so below we will need to check for coordinates that fall in the padding area.

        size_t n_spatial_dimensions = arg_shape.size() - 2;

        Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
        Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
        Strides input_batch_transform_source_strides(2 + n_spatial_dimensions, 1);
        AxisVector input_batch_transform_source_axis_order(2 + n_spatial_dimensions);
        CoordinateDiff input_batch_transform_padding_below(2 + n_spatial_dimensions);
        CoordinateDiff input_batch_transform_padding_above(2 + n_spatial_dimensions);

        input_batch_transform_start[0] = batch_index;
        input_batch_transform_end[0] = batch_index + 1;
        input_batch_transform_start[1] = channel;
        input_batch_transform_end[1] = channel + 1;
        input_batch_transform_padding_below[0] = 0;
        input_batch_transform_padding_below[1] = 0;
        input_batch_transform_padding_above[0] = 0;
        input_batch_transform_padding_above[1] = 0;

        for (size_t i = 2; i < n_spatial_dimensions + 2; i++)
        {
size_t window_shape_this_dim = window_shape[i - 2];
size_t movement_stride = window_movement_strides[i - 2];

input_batch_transform_start[i] = movement_stride * out_coord[i];
input_batch_transform_end[i] =
    input_batch_transform_start[i] + window_shape_this_dim;
input_batch_transform_padding_below[i] = padding_below[i - 2];
input_batch_transform_padding_above[i] = padding_above[i - 2];
        }

        for (size_t i = 0; i < arg_shape.size(); i++)
        {
input_batch_transform_source_axis_order[i] = i;
        }

        CoordinateTransform input_batch_transform(
arg_shape,
input_batch_transform_start,
input_batch_transform_end,
input_batch_transform_source_strides,
input_batch_transform_source_axis_order,
input_batch_transform_padding_below,
input_batch_transform_padding_above);

        // As we go, we compute the maximum value:
        //
        //   output[O] = max(output[O],arg[I])

        T result = std::numeric_limits<T>::lowest();

        for (const Coordinate input_batch_coord : input_batch_transform)
        {
if (input_batch_transform.has_source_coordinate(input_batch_coord))
{
    T x = arg[input_batch_transform.index(input_batch_coord)];
    result = x > result ? x : result;
}
        }

        out[output_transform.index(out_coord)] = result;
    }
}
)");
LU_DEFINE(cpu_reference_reverse, R"(template <typename T>
void cpu_reference_reverse(const T* arg,
 T* out,
 const Shape arg_shape,
 const Shape out_shape,
 const AxisSet reversed_axes)
{
    // In fact arg_shape == out_shape, but we'll use both for stylistic consistency with other kernels.
    CoordinateTransform arg_transform(arg_shape);
    CoordinateTransform output_transform(out_shape);

    for (Coordinate out_coord : output_transform)
    {
        Coordinate arg_coord = out_coord;

        for (size_t i = 0; i < arg_coord.size(); i++)
        {
if (reversed_axes.count(i) != 0)
{
    arg_coord[i] = arg_shape[i] - arg_coord[i] - 1;
}
        }

        out[output_transform.index(out_coord)] = arg[arg_transform.index(arg_coord)];
    }
}
)");
LU_DEFINE(cpu_reference_acos, R"(template <typename T>
void cpu_reference_acos(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::cpu_reference_acos(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_exp, R"(template <typename T>
void cpu_reference_exp(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::cpu_reference_exp(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_generate_mask, R"(template <typename T>
void cpu_reference_generate_mask(T* out, size_t count, ngraph::RNGState* rng_state, bool training)
{
    auto& gen = rng_state->get_generator();
    auto& bd = rng_state->get_distribution();

    for (size_t i = 0; i < count; i++)
    {
        out[i] = training ? static_cast<T>(bd(gen)) : static_cast<T>(1);
    }
}
)");
LU_DEFINE(cpu_reference_dot, R"(template <typename T>
void cpu_reference_dot(const T* arg0,
         const T* arg1,
         T* out,
         const Shape arg0_shape,
         const Shape arg1_shape,
         const Shape out_shape,
         size_t reduction_axes_count)
{
    // Get the sizes of the dot axes. It's easiest to pull them from arg1 because they're
    // right up front.
    Shape dot_axis_sizes(reduction_axes_count);
    std::copy(arg1_shape.begin(),
  arg1_shape.begin() + reduction_axes_count,
  dot_axis_sizes.begin());

    CoordinateTransform arg0_transform(arg0_shape);
    CoordinateTransform arg1_transform(arg1_shape);
    CoordinateTransform output_transform(out_shape);

    // Create coordinate transforms for arg0 and arg1 that throw away the dotted axes.
    size_t arg0_projected_rank = arg0_shape.size() - reduction_axes_count;
    size_t arg1_projected_rank = arg1_shape.size() - reduction_axes_count;

    Shape arg0_projected_shape(arg0_projected_rank);
    std::copy(arg0_shape.begin(),
  arg0_shape.begin() + arg0_projected_rank,
  arg0_projected_shape.begin());

    Shape arg1_projected_shape(arg1_projected_rank);
    std::copy(arg1_shape.begin() + reduction_axes_count,
  arg1_shape.end(),
  arg1_projected_shape.begin());

    CoordinateTransform arg0_projected_transform(arg0_projected_shape);
    CoordinateTransform arg1_projected_transform(arg1_projected_shape);

    // Create a coordinate transform that allows us to iterate over all possible values
    // for the dotted axes.
    CoordinateTransform dot_axes_transform(dot_axis_sizes);

    for (const Coordinate arg0_projected_coord : arg0_projected_transform)
    {
        for (const Coordinate arg1_projected_coord : arg1_projected_transform)
        {
// The output coordinate is just the concatenation of the projected coordinates.
Coordinate out_coord(arg0_projected_coord.size() +
         arg1_projected_coord.size());

auto out_coord_it = std::copy(arg0_projected_coord.begin(),
      arg0_projected_coord.end(),
      out_coord.begin());
std::copy(
    arg1_projected_coord.begin(), arg1_projected_coord.end(), out_coord_it);

// Zero out to start the sum.
T sum = 0;

size_t out_index = output_transform.index(out_coord);

// Walk along the dotted axes.
Coordinate arg0_coord(arg0_shape.size());
Coordinate arg1_coord(arg1_shape.size());
auto arg0_it = std::copy(arg0_projected_coord.begin(),
 arg0_projected_coord.end(),
 arg0_coord.begin());
for (const Coordinate dot_axis_positions : dot_axes_transform)
{
    // In order to find the points to multiply together, we need to inject our current
    // positions along the dotted axes back into the projected arg0 and arg1 coordinates.
    std::copy(
        dot_axis_positions.begin(), dot_axis_positions.end(), arg0_it);

    auto arg1_it = std::copy(dot_axis_positions.begin(),
     dot_axis_positions.end(),
     arg1_coord.begin());
    std::copy(
        arg1_projected_coord.begin(), arg1_projected_coord.end(), arg1_it);

    // Multiply and add to the sum.
    sum += arg0[arg0_transform.index(arg0_coord)] *
           arg1[arg1_transform.index(arg1_coord)];
}

// Write the sum back.
out[out_index] = sum;
        }
    }
}
)");
LU_DEFINE(cpu_reference_maximum, R"(template <typename T>
void cpu_reference_maximum(const T* arg0, const T* arg1, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i] > arg1[i] ? arg0[i] : arg1[i];
    }
}
)");
LU_DEFINE(cpu_reference_minimum, R"(template <typename T>
void cpu_reference_minimum(const T* arg0, const T* arg1, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i] < arg1[i] ? arg0[i] : arg1[i];
    }
}
)");
LU_DEFINE(cpu_reference_greater_eq, R"(template <typename T>
void cpu_reference_greater_eq(const T* arg0,
    const T* arg1,
    char* out,
    size_t count) // TODO: using char for bool, is this right?
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i] >= arg1[i];
    }
}
)");
LU_DEFINE(cpu_reference_avg_pool, R"(template <typename T>
void avg_pool_backprop(const T* delta,
           T* out,
           const Shape delta_shape,
           const Shape out_shape,
           const Shape window_shape,
           const Strides window_movement_strides,
           const Shape padding_below,
           const Shape padding_above,
           bool include_padding_in_avg_computation)
{
    CoordinateTransform out_transform(out_shape);

    for (const Coordinate out_coord : out_transform)
    {
        out[out_transform.index(out_coord)] = 0;
    }

    CoordinateTransform delta_transform(delta_shape);

    for (const Coordinate delta_coord : delta_transform)
    {
        size_t img_index = delta_coord[0];
        size_t channel = delta_coord[1];

        size_t n_image_dimensions = out_shape.size() - 2;
        Coordinate source_window_transform_start(2 + n_image_dimensions);
        Coordinate source_window_transform_end(2 + n_image_dimensions);
        Strides source_window_transform_source_strides(2 + n_image_dimensions, 1);
        AxisVector source_window_transform_source_axis_order(2 + n_image_dimensions);
        CoordinateDiff source_window_transform_padding_below(2 + n_image_dimensions);
        CoordinateDiff source_window_transform_padding_above(2 + n_image_dimensions);

        source_window_transform_start[0] = img_index;
        source_window_transform_end[0] = img_index + 1;
        source_window_transform_start[1] = channel;
        source_window_transform_end[1] = channel + 1;
        source_window_transform_padding_below[0] = 0;
        source_window_transform_padding_below[1] = 0;
        source_window_transform_padding_above[0] = 0;
        source_window_transform_padding_above[1] = 0;

        for (size_t i = 2; i < n_image_dimensions + 2; i++)
        {
size_t window_shape_this_dim = window_shape[i - 2];
size_t movement_stride = window_movement_strides[i - 2];

source_window_transform_start[i] = movement_stride * delta_coord[i];
source_window_transform_end[i] =
    source_window_transform_start[i] + window_shape_this_dim;
source_window_transform_padding_below[i] = padding_below[i - 2];
source_window_transform_padding_above[i] = padding_above[i - 2];
        }
        std::iota(begin(source_window_transform_source_axis_order),
      end(source_window_transform_source_axis_order),
      0);

        CoordinateTransform source_window_transform(
out_shape,
source_window_transform_start,
source_window_transform_end,
source_window_transform_source_strides,
source_window_transform_source_axis_order,
source_window_transform_padding_below,
source_window_transform_padding_above);

        size_t num_elements_in_window = 0;

        for (const Coordinate source_window_coord : source_window_transform)
        {
if (source_window_transform.has_source_coordinate(source_window_coord) ||
    include_padding_in_avg_computation)
{
    num_elements_in_window++;
}
        }

        for (const Coordinate source_window_coord : source_window_transform)
        {
if (source_window_transform.has_source_coordinate(source_window_coord))
{
    size_t out_index = source_window_transform.index(source_window_coord);
    out[out_index] +=
        delta[delta_transform.index(delta_coord)] / num_elements_in_window;
}
        }
    }
}

template <typename T>
void cpu_reference_avg_pool(const T* arg,
  T* out,
  const Shape arg_shape,
  const Shape out_shape,
  const Shape window_shape,
  const Strides window_movement_strides,
  const Shape padding_below,
  const Shape padding_above,
  bool include_padding_in_avg_computation)
{
    // At the outermost level we will walk over every output coordinate O.
    CoordinateTransform output_transform(out_shape);

    for (const Coordinate out_coord : output_transform)
    {
        // Our output coordinate O will have the form:
        //
        //   (N,chan,i_1,...,i_n)

        size_t batch_index = out_coord[0];
        size_t channel = out_coord[1];

        // For the input data we need to iterate the coordinate:
        //
        //   I:
        //
        // over the range (noninclusive on the right):
        //
        //   (N,chan,s_1*i_1,s_2*i_2,...,s_n*i_n) ->
        //
        //     (N+1,chan+1,s_1*i_1 + window_shape_1,...,s_n*i_n + window_shape_n)
        //
        // with unit stride.
        //
        // We iterate this over the *padded* data, so below we will need to check for coordinates that fall in the padding area.

        size_t n_spatial_dimensions = arg_shape.size() - 2;

        Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
        Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
        Strides input_batch_transform_source_strides(2 + n_spatial_dimensions, 1);
        AxisVector input_batch_transform_source_axis_order(2 + n_spatial_dimensions);
        CoordinateDiff input_batch_transform_padding_below(2 + n_spatial_dimensions);
        CoordinateDiff input_batch_transform_padding_above(2 + n_spatial_dimensions);

        input_batch_transform_start[0] = batch_index;
        input_batch_transform_end[0] = batch_index + 1;
        input_batch_transform_start[1] = channel;
        input_batch_transform_end[1] = channel + 1;
        input_batch_transform_padding_below[0] = 0;
        input_batch_transform_padding_below[1] = 0;
        input_batch_transform_padding_above[0] = 0;
        input_batch_transform_padding_above[1] = 0;

        for (size_t i = 2; i < n_spatial_dimensions + 2; i++)
        {
size_t window_shape_this_dim = window_shape[i - 2];
size_t movement_stride = window_movement_strides[i - 2];

input_batch_transform_start[i] = movement_stride * out_coord[i];
input_batch_transform_end[i] =
    input_batch_transform_start[i] + window_shape_this_dim;
input_batch_transform_padding_below[i] = padding_below[i - 2];
input_batch_transform_padding_above[i] = padding_above[i - 2];
        }

        for (size_t i = 0; i < arg_shape.size(); i++)
        {
input_batch_transform_source_axis_order[i] = i;
        }

        CoordinateTransform input_batch_transform(
arg_shape,
input_batch_transform_start,
input_batch_transform_end,
input_batch_transform_source_strides,
input_batch_transform_source_axis_order,
input_batch_transform_padding_below,
input_batch_transform_padding_above);

        // As we go, we compute the sum value:
        //
        //   output[O] := output[O] + arg[I]
        //
        // and the number of elements:
        //
        //   n_elements := n_elements + 1

        T result = 0;
        size_t n_elements = 0;

        for (const Coordinate input_batch_coord : input_batch_transform)
        {
bool in_bounds =
    input_batch_transform.has_source_coordinate(input_batch_coord);

if (in_bounds || include_padding_in_avg_computation)
{
    T v =
        in_bounds ? arg[input_batch_transform.index(input_batch_coord)] : 0;
    result += v;
    n_elements++;
}
        }

        if (n_elements == 0)
        {
throw std::runtime_error("AvgPool elements == 0, must be non-zero");
        }

        out[output_transform.index(out_coord)] = result / n_elements;
    }
}
)");
LU_DEFINE(cpu_reference_softmax, R"(template <typename T>
void cpu_reference_softmax(const T* arg, T* out, const Shape shape, const AxisSet axes)
{
    auto temp_shape = reduce(shape, axes);
    auto temp_elements = std::accumulate(
        temp_shape.begin(), temp_shape.end(), 1, std::multiplies<size_t>());
    auto temp_ptr = new T[temp_elements];

    cpu_reference_max(arg, temp_ptr, shape, temp_shape, axes);

    CoordinateTransform transform(shape);
    CoordinateTransform temp_transform(temp_shape);
    for (const Coordinate coord : transform)
    {
        Coordinate temp_coord = reduce(coord, axes);
        out[transform.index(coord)] = std::exp(
arg[transform.index(coord)] - temp_ptr[temp_transform.index(temp_coord)]);
    }

    cpu_reference_sum(out, temp_ptr, shape, temp_shape, axes);

    for (const Coordinate coord : transform)
    {
        Coordinate temp_coord = reduce(coord, axes);
        out[transform.index(coord)] /= temp_ptr[temp_transform.index(temp_coord)];
    }

    delete[] temp_ptr;
}
)");
LU_DEFINE(
    cpu_reference_divide,
    R"(// NOTE: Execution throws `std::domain_error` if either a non-integral value or an out-of-bounds
// value is detected in the input tensor.

// In English: return type is void and T must be an integral type.
template <typename T>
typename std::enable_if<std::is_integral<T>::value>::type
    cpu_reference_divide(const T* arg0, const T* arg1, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        if (arg1[i] == 0)
        {
throw std::domain_error("integer division by zero");
        }
        out[i] = arg0[i] / arg1[i];
    }
}

// In English: return type is void and T must be a floating point type.
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value>::type
    cpu_reference_divide(const T* arg0, const T* arg1, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        // TODO: Here we do not check for div by zero, so we'll get +-inf here
        // if arg1[i] == 0. Is that the right thing to do? Jury's still out.
        out[i] = arg0[i] / arg1[i];
    }
}
)");
LU_DEFINE(cpu_reference_negate, R"(template <typename T>
void cpu_reference_negate(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = -arg[i];
    }
}
)");
LU_DEFINE(cpu_reference_select, R"(template <typename T>
void cpu_reference_select(const char* arg0,
const T* arg1,
const T* arg2,
T* out,
size_t count) // TODO: using char for bool, is this right?
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i] ? arg1[i] : arg2[i];
    }
}
)");
LU_DEFINE(cpu_reference_sqrt, R"(template <typename T>
void cpu_reference_sqrt(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::sqrt(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_atan, R"(template <typename T>
void cpu_reference_atan(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::cpu_reference_atan(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_tanh, R"(template <typename T>
void cpu_reference_tanh(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::cpu_reference_tanh(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_sin, R"(template <typename T>
void cpu_reference_sin(const T* arg, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = std::cpu_reference_sin(arg[i]);
    }
}
)");
LU_DEFINE(cpu_reference_subtract, R"(template <typename T>
void cpu_reference_subtract(const T* arg0, const T* arg1, T* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = arg0[i] - arg1[i];
    }
}
)");
LU_DEFINE(cpu_reference_slice, R"(template <typename T>
void cpu_reference_slice(const T* arg,
           T* out,
           const Shape arg_shape,
           const Coordinate lower_bounds,
           const Coordinate upper_bounds,
           const Strides strides,
           const Shape out_shape)
{
    CoordinateTransform input_transform(arg_shape, lower_bounds, upper_bounds, strides);
    CoordinateTransform output_transform(out_shape);

    CoordinateTransform::Iterator output_it = output_transform.begin();

    assert(shape_size(input_transform.get_target_shape()) ==
      shape_size(output_transform.get_target_shape()));

    for (const Coordinate in_coord : input_transform)
    {
        const Coordinate out_coord = *output_it;

        out[output_transform.index(out_coord)] = arg[input_transform.index(in_coord)];

        ++output_it;
    }
}
)");
LU_DEFINE(cpu_reference_replace_slice, R"(template <typename T>
void cpu_reference_replace_slice(const T* arg0, // replacement context
       const T* arg1, // replacement value
       T* out,
       const Shape arg1_shape,
       const Coordinate lower_bounds,
       const Coordinate upper_bounds,
       const Strides strides,
       const Shape out_shape)
{
    // Step 1: Copy the entire replacement context to the output.
    CoordinateTransform copy_transform(out_shape);

    for (Coordinate copy_coord : copy_transform)
    {
        out[copy_transform.index(copy_coord)] = arg0[copy_transform.index(copy_coord)];
    }

    // Step 2: Overwrite the slice for replacement.
    CoordinateTransform input_transform(arg1_shape);
    CoordinateTransform output_transform(
        out_shape, lower_bounds, upper_bounds, strides);

    assert(shape_size(input_transform.get_target_shape()) ==
      shape_size(output_transform.get_target_shape()));

    CoordinateTransform::Iterator output_it = output_transform.begin();

    for (const Coordinate input_coord : input_transform)
    {
        const Coordinate output_coord = *output_it;

        out[output_transform.index(output_coord)] =
arg1[input_transform.index(input_coord)];

        ++output_it;
    }
}
)");
LU_DEFINE(
    cpu_reference_one_hot,
    R"(// NOTE: Execution throws `std::range_error` if either a non-integral value or an out-of-bounds
// value is detected in the input tensor.
template <typename T>
void cpu_reference_one_hot(const T* arg,
 T* out,
 const Shape in_shape,
 const Shape out_shape,
 size_t one_hot_axis)
{
    // Step 1: Zero out the output.
    CoordinateTransform output_transform(out_shape);

    for (const Coordinate output_coord : output_transform)
    {
        out[output_transform.index(output_coord)] = 0;
    }

    // Step 2: Write ones at needed positions, throwing exceptions when invalid conditions
    // are encountered.
    CoordinateTransform input_transform(in_shape);

    for (const Coordinate input_coord : input_transform)
    {
        T val = arg[input_transform.index(input_coord)];

        if (std::floor(val) < val || std::floor(val) > val)
        {
throw(std::range_error("One-hot: non-integral value in input"));
        }

        size_t one_hot_pos = static_cast<size_t>(val);

        if (one_hot_pos >= out_shape[one_hot_axis])
        {
throw(std::range_error("One-hot: value is out of category range"));
        }

        Coordinate one_hot_coord = inject(input_coord, one_hot_axis, one_hot_pos);

        out[output_transform.index(one_hot_coord)] = 1;
    }
}
)");

//Classes
namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class AbsRef : public KernelEmitter
            {
            public:
                AbsRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Abs>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_abs<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_abs);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Abs> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Abs",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                AbsRef)                                                            // constructor

            class AcosRef : public KernelEmitter
            {
            public:
                AcosRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Acos>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_acos<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_acos);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Acos> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Acos",                                                            // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                AcosRef)                                                           // constructor

            class AddRef : public KernelEmitter
            {
            public:
                AddRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Add>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << ctx->gnode->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_add<" << dtype.c_type_string()
                       << ">(input0,input1,output0," << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_add);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Add> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Add",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                AddRef)                                                            // constructor

            class AllReduceRef : public KernelEmitter
            {
            public:
                AllReduceRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::AllReduce>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_allreduce_float(input0,output0,static_cast<int>("
                       << m_context->inputs[0]->size(false) << "));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_allreduce);
                    lu.require(header_mpi);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::AllReduce> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "AllReduce",                                                       // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                AllReduceRef)                                                      // constructor

            class AsinRef : public KernelEmitter
            {
            public:
                AsinRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Asin>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_asin<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_asin);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Asin> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Asin",                                                            // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                AsinRef)                                                           // constructor

            class AtanRef : public KernelEmitter
            {
            public:
                AtanRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Atan>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_atan<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_atan);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Atan> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Atan",                                                            // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                AtanRef)                                                           // constructor

            class BroadcastRef : public KernelEmitter
            {
            public:
                BroadcastRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Broadcast>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_broadcast<" << dtype.c_type_string()
                       << ">(input0,output0,Shape({" << join(m_context->inputs[0]->get_shape())
                       << "}),Shape({" << join(m_context->outputs[0]->get_shape()) << "}),AxisSet({"
                       << join(op->get_broadcast_axes()) << "}));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_broadcast);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Broadcast> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Broadcast",                                                       // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                BroadcastRef)                                                      // constructor

            class CeilingRef : public KernelEmitter
            {
            public:
                CeilingRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Ceiling>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_ceiling<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_ceiling);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Ceiling> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Ceiling",                                                         // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                CeilingRef)                                                        // constructor

            class ConcatRef : public KernelEmitter
            {
            public:
                ConcatRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Concat>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "std::vector<const float*> in_args;";
                    lu << "std::vector<Shape> in_shapes;";
                    for (size_t t = 0; t < m_context->inputs.size(); t++)
                    {
                        lu << "in_args.push_back( input" << t << ");";
                        lu << "in_shapes.push_back(Shape({"
                           << join(m_context->inputs[t]->get_shape()) << "}));";
                    }
                    lu << "cpu_reference_concat<" << dtype.c_type_string()
                       << ">(in_args,output0, in_shapes,Shape({"
                       << join(m_context->outputs[0]->get_shape()) << "}), "
                       << op->get_concatenation_axis() << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_concat);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Concat> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Concat",                                                          // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                ConcatRef)                                                         // constructor

            /*
            class ConstantRef : public KernelEmitter
            {
            public:
                ConstantRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Constant>(ctx->gnode->get_op_ptr()); std::stringstream tag; tag << op->get_name(); custom_tag = tag.str();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    // lu<<"cpu_reference_constant<float>(OP->get_data_ptr<float>(),output0,"<<m_context->outputs[0]->size(false)<<");";
                    lu << "//<todo> left blank by purpose.";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_constant);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Constant> op;
            };

            REGISTER_KERNEL_EMITTER(
                "Constant",                                                    // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                ConstantRef)                                                   // constructor
            */

            class ConvertRef : public KernelEmitter
            {
            public:
                ConvertRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Convert>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    auto in_type = m_context->inputs[0]->get_element_type().c_type_string();
                    auto out_type = m_context->outputs[0]->get_element_type().c_type_string();
                    lu << "cpu_reference_convert<" << in_type << ", " << out_type
                       << ">(input0, output0," << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_convert);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Convert> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Convert",                                                         // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                ConvertRef)                                                        // constructor

            class ConvolutionRef : public KernelEmitter
            {
            public:
                ConvolutionRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Convolution>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_convolution<" << dtype.c_type_string()
                       << ">(input0,input1,output0,Shape({"
                       << join(m_context->inputs[0]->get_shape()) << "}),Shape({"
                       << join(m_context->inputs[1]->get_shape()) << "}),Shape({"
                       << join(m_context->outputs[0]->get_shape()) << "}),Strides({"
                       << join(op->get_window_movement_strides()) << "}),Strides({"
                       << join(op->get_window_dilation_strides()) << "}),CoordinateDiff({"
                       << join(op->get_padding_below()) << "}),CoordinateDiff({"
                       << join(op->get_padding_above()) << "}),Strides({"
                       << join(op->get_data_dilation_strides()) << "}),0,1,1,0,0,1,false);";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_convolution);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Convolution> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Convolution",                                                     // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                ConvolutionRef)                                                    // constructor

            class CosRef : public KernelEmitter
            {
            public:
                CosRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Cos>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_cos<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_cos);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Cos> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Cos",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                CosRef)                                                            // constructor

            class CoshRef : public KernelEmitter
            {
            public:
                CoshRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Cosh>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_cosh<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_cosh);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Cosh> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Cosh",                                                            // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                CoshRef)                                                           // constructor

            class DivideRef : public KernelEmitter
            {
            public:
                DivideRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Divide>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_divide<" << dtype.c_type_string()
                       << ">(input0,input1,output0," << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_divide);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Divide> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Divide",                                                          // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                DivideRef)                                                         // constructor

            class EqualRef : public KernelEmitter
            {
            public:
                EqualRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Equal>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_equal<" << dtype.c_type_string()
                       << ">(input0,input1,(char*)output0," << m_context->outputs[0]->size(false)
                       << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_equal);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Equal> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Equal",                                                           // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                EqualRef)                                                          // constructor

            class ExpRef : public KernelEmitter
            {
            public:
                ExpRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Exp>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_exp<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_exp);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Exp> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Exp",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                ExpRef)                                                            // constructor

            class FloorRef : public KernelEmitter
            {
            public:
                FloorRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Floor>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_floor<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_floor);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Floor> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Floor",                                                           // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                FloorRef)                                                          // constructor

            class GreaterRef : public KernelEmitter
            {
            public:
                GreaterRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Greater>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_greater<" << dtype.c_type_string()
                       << ">(input0,input1,(char*)output0," << m_context->outputs[0]->size(false)
                       << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_greater);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Greater> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Greater",                                                         // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                GreaterRef)                                                        // constructor

            class LessRef : public KernelEmitter
            {
            public:
                LessRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Less>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_less<" << dtype.c_type_string()
                       << ">(input0,input1,(char*)output0," << m_context->outputs[0]->size(false)
                       << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_less);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Less> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Less",                                                            // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                LessRef)                                                           // constructor

            class LogRef : public KernelEmitter
            {
            public:
                LogRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Log>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_log<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_log);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Log> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Log",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                LogRef)                                                            // constructor

            class LRNRef : public KernelEmitter
            {
            public:
                LRNRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::LRN>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_lrn<" << dtype.c_type_string()
                       << ">(input0,output0,Shape({" << join(m_context->inputs[0]->get_shape())
                       << "})," << (op->get_alpha()) << "," << (op->get_beta()) << ","
                       << (op->get_bias()) << "," << (op->get_nsize()) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_lrn);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::LRN> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "LRN",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                LRNRef)                                                            // constructor

            class MaxRef : public KernelEmitter
            {
            public:
                MaxRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Max>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_max<" << dtype.c_type_string()
                       << ">(input0,output0,Shape({" << join(m_context->inputs[0]->get_shape())
                       << "}),Shape({" << join(m_context->outputs[0]->get_shape()) << "}),AxisSet({"
                       << join(op->get_reduction_axes()) << "}));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_max);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Max> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Max",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                MaxRef)                                                            // constructor

            class MaximumRef : public KernelEmitter
            {
            public:
                MaximumRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Maximum>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_maximum<" << dtype.c_type_string()
                       << ">(input0,input1,output0," << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_maximum);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Maximum> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Maximum",                                                         // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                MaximumRef)                                                        // constructor

            class MinRef : public KernelEmitter
            {
            public:
                MinRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Min>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_min<" << dtype.c_type_string()
                       << ">(input0,output0,Shape({" << join(m_context->inputs[0]->get_shape())
                       << "}),Shape({" << join(m_context->outputs[0]->get_shape()) << "}),AxisSet({"
                       << join(op->get_reduction_axes()) << "}));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_min);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Min> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Min",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                MinRef)                                                            // constructor

            class MinimumRef : public KernelEmitter
            {
            public:
                MinimumRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Minimum>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_minimum<" << dtype.c_type_string()
                       << ">(input0,input1,output0," << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_minimum);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Minimum> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Minimum",                                                         // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                MinimumRef)                                                        // constructor

            class MultiplyRef : public KernelEmitter
            {
            public:
                MultiplyRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Multiply>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_multiply<" << dtype.c_type_string()
                       << ">(input0,input1,output0," << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_multiply);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Multiply> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Multiply",                                                        // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                MultiplyRef)                                                       // constructor

            class NegativeRef : public KernelEmitter
            {
            public:
                NegativeRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Negative>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_negate<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_negate);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Negative> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Negative",                                                        // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                NegativeRef)                                                       // constructor

            class PowerRef : public KernelEmitter
            {
            public:
                PowerRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Power>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_power<" << dtype.c_type_string()
                       << ">(input0,input1,output0," << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_power);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Power> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Power",                                                           // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                PowerRef)                                                          // constructor

            class ProductRef : public KernelEmitter
            {
            public:
                ProductRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Product>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_product<" << dtype.c_type_string()
                       << ">(input0,output0,Shape({" << join(m_context->inputs[0]->get_shape())
                       << "}),Shape({" << join(m_context->outputs[0]->get_shape()) << "}),AxisSet({"
                       << join(op->get_reduction_axes()) << "}));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_product);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Product> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Product",                                                         // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                ProductRef)                                                        // constructor

            class ReluRef : public KernelEmitter
            {
            public:
                ReluRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Relu>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_relu<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_relu);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Relu> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Relu",                                                            // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                ReluRef)                                                           // constructor

            class SelectRef : public KernelEmitter
            {
            public:
                SelectRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Select>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_select<" << dtype.c_type_string()
                       << ">(args[0]->get_data_ptr<char>(),input1,input2,"
                          "output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_select);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Select> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Select",                                                          // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                SelectRef)                                                         // constructor

            class SigmoidRef : public KernelEmitter
            {
            public:
                SigmoidRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Sigmoid>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_sigmoid<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_sigmoid);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Sigmoid> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Sigmoid",                                                         // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                SigmoidRef)                                                        // constructor

            class SignRef : public KernelEmitter
            {
            public:
                SignRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Sign>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_sign<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_sign);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Sign> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Sign",                                                            // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                SignRef)                                                           // constructor

            class SinRef : public KernelEmitter
            {
            public:
                SinRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Sin>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_sin<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_sin);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Sin> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Sin",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                SinRef)                                                            // constructor

            class SinhRef : public KernelEmitter
            {
            public:
                SinhRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Sinh>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_sinh<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_sinh);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Sinh> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Sinh",                                                            // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                SinhRef)                                                           // constructor

            class SliceRef : public KernelEmitter
            {
            public:
                SliceRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Slice>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_slice<" << dtype.c_type_string()
                       << ">(input0,output0,Shape({" << join(m_context->inputs[0]->get_shape())
                       << "}),Coordinate({" << join(op->get_lower_bounds()) << "}),Coordinate({"
                       << join(op->get_upper_bounds()) << "}),Strides({" << join(op->get_strides())
                       << "}),Shape({" << join(m_context->outputs[0]->get_shape()) << "}));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_slice);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Slice> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Slice",                                                           // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                SliceRef)                                                          // constructor

            class SoftmaxRef : public KernelEmitter
            {
            public:
                SoftmaxRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Softmax>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_softmax<" << dtype.c_type_string()
                       << ">(input0,output0,Shape({" << join(m_context->outputs[0]->get_shape())
                       << "}), AxisSet({" << join(op->get_axes()) << "}));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_max);
                    lu.require(cpu_reference_sum);
                    lu.require(cpu_reference_softmax);
                    //<todo> migerate those into definition
                    // cpu_reference_softmax->require(cpu_reference_max);
                    // cpu_reference_softmax->require(cpu_reference_sum);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Softmax> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Softmax",                                                         // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                SoftmaxRef)                                                        // constructor

            class SqrtRef : public KernelEmitter
            {
            public:
                SqrtRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Sqrt>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_sqrt<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_sqrt);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Sqrt> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Sqrt",                                                            // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                SqrtRef)                                                           // constructor

            class SubtractRef : public KernelEmitter
            {
            public:
                SubtractRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Subtract>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_subtract<" << dtype.c_type_string()
                       << ">(input0,input1,output0," << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_subtract);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Subtract> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Subtract",                                                        // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                SubtractRef)                                                       // constructor

            class SumRef : public KernelEmitter
            {
            public:
                SumRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Sum>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_sum<" << dtype.c_type_string()
                       << ">(input0,output0,Shape({" << join(m_context->inputs[0]->get_shape())
                       << "}),Shape({" << join(m_context->outputs[0]->get_shape()) << "}),AxisSet({"
                       << join(op->get_reduction_axes()) << "}));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_sum);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Sum> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Sum",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                SumRef)                                                            // constructor

            class TanRef : public KernelEmitter
            {
            public:
                TanRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Tan>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_tan<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_tan);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Tan> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Tan",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                TanRef)                                                            // constructor

            class TanhRef : public KernelEmitter
            {
            public:
                TanhRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Tanh>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_tanh<" << dtype.c_type_string() << ">(input0,output0,"
                       << m_context->outputs[0]->size(false) << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_tanh);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Tanh> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Tanh",                                                            // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                TanhRef)                                                           // constructor

            class BatchNormRef : public KernelEmitter
            {
            public:
                BatchNormRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::BatchNormInference>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_batch_norm<" << dtype.c_type_string() << ">("
                       << op->get_eps_value() << ","
                       << "input0, input1, input2, input3, input4, output0, "
                       << "Shape({" << join(m_context->inputs[2]->get_shape()) << "}));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_batch_norm);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::BatchNormInference> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "BatchNormInference",                                              // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                BatchNormRef)

            class AvgPoolRef : public KernelEmitter
            {
            public:
                AvgPoolRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::AvgPool>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_avg_pool<" << dtype.c_type_string() << ">(input0,output0,"
                       << "Shape({" << join(m_context->inputs[0]->get_shape()) << "}),"
                       << "Shape({" << join(m_context->outputs[0]->get_shape()) << "}),"
                       << "Shape({" << join(op->get_window_shape()) << "}),"
                       << "Strides({" << join(op->get_window_movement_strides()) << "}),"
                       << "Shape({" << join(op->get_padding_below()) << "}),"
                       << "Shape({" << join(op->get_padding_above()) << "}),"
                       << op->get_include_padding_in_avg_computation() << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_avg_pool);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::AvgPool> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "AvgPool",                                                         // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                AvgPoolRef)

            class DotRef : public KernelEmitter
            {
            public:
                DotRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Dot>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_dot<" << dtype.c_type_string()
                       << ">(input0,input1,output0,"
                       << "Shape({" << join(m_context->inputs[0]->get_shape()) << "}),"
                       << "Shape({" << join(m_context->inputs[1]->get_shape()) << "}),"
                       << "Shape({" << join(m_context->outputs[0]->get_shape()) << "}),"
                       << op->get_reduction_axes_count() << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_dot);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Dot> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Dot",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                DotRef)

            class MaxPoolRef : public KernelEmitter
            {
            public:
                MaxPoolRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::MaxPool>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_max_pool<" << dtype.c_type_string() << ">(input0,output0,"
                       << "Shape({" << join(m_context->inputs[0]->get_shape()) << "}),"
                       << "Shape({" << join(m_context->outputs[0]->get_shape()) << "}),"
                       << "Shape({" << join(op->get_window_shape()) << "}),"
                       << "Strides({" << join(op->get_window_movement_strides()) << "}),"
                       << "Shape({" << join(op->get_padding_below()) << "}),"
                       << "Shape({" << join(op->get_padding_above()) << "}));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_max_pool);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::MaxPool> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "MaxPool",                                                         // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                MaxPoolRef)

            class PadRef : public KernelEmitter
            {
            public:
                PadRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Pad>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_pad<" << dtype.c_type_string()
                       << ">(input0, input1, output0,"
                       << "Shape({" << join(m_context->inputs[0]->get_shape()) << "}),"
                       << "Shape({" << join(m_context->outputs[0]->get_shape()) << "}),"
                       << "Shape({" << join(op->get_padding_below()) << "}),"
                       << "Shape({" << join(op->get_padding_above()) << "}),"
                       << "Shape({" << join(op->get_padding_interior()) << "}));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_pad);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Pad> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Pad",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                PadRef)

            class ReshapeRef : public KernelEmitter
            {
            public:
                ReshapeRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Reshape>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_reshape<" << dtype.c_type_string() << ">(input0, output0,"
                       << "Shape({" << join(m_context->inputs[0]->get_shape()) << "}),"
                       << "AxisVector({" << join(op->get_input_order()) << "}),"
                       << "Shape({" << join(m_context->outputs[0]->get_shape()) << "}));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_reshape);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Reshape> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Reshape",                                                         // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                ReshapeRef)

            class ResultRef : public KernelEmitter
            {
            public:
                ResultRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Result>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());

                    if (FLAGS_fextern_result_memory)
                    {
                        lu << "cpu_reference_result<" << dtype.c_type_string()
                           << ">(input0, output0," << shape_size(m_context->outputs[0]->get_shape())
                           << ");";
                    }
                    else
                    {
                        lu << "*output0 = input0;";
                    }
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_result);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Result> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Result",                                                          // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                ResultRef)

            class LessEqRef : public KernelEmitter
            {
            public:
                LessEqRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::LessEq>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_less_eq<" << dtype.c_type_string()
                       << ">(input0,input1,(char*)output0," << m_context->outputs[0]->size(false)
                       << ");";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_less_eq);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::LessEq> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "LessEq",                                                          // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                LessEqRef)

            class ReverseRef : public KernelEmitter
            {
            public:
                ReverseRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "reference")
                {
                    op = static_pointer_cast<op::Reverse>(ctx->gnode->get_op_ptr());
                    std::stringstream tag;
                    tag << op->get_name();
                    custom_tag = tag.str();
                    dtype = ctx->outputs[0]->get_element_type();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit lu(get_function_name());
                    lu << "cpu_reference_reverse<" << dtype.c_type_string() << ">(input0,output0,"
                       << "Shape({" << join(m_context->inputs[0]->get_shape()) << "}),"
                       << "Shape({" << join(m_context->outputs[0]->get_shape()) << "}),"
                       << "AxisSet({" << join(op->get_reversed_axes()) << "}));";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    lu.require(cpu_reference_common);
                    lu.require(cpu_reference_reverse);
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<op::Reverse> op;
                element::Type dtype;
            };

            REGISTER_KERNEL_EMITTER(
                "Reverse",                                                         // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                ReverseRef)

        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
