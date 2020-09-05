// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Example: IMAGES=/opt/images/train/ ./image_test

#include <stdio.h>
#define __IMAGE_TESTS__

#include <assert.h>
#include <dirent.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <stdint.h>
#include <string.h>

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "nnfusion_rt.h"

static void WalkFiles(const std::string& path,
                      const std::function<void(const std::string& file)>& walker)
{
    if (path.size() >= 0 && path.back() != '/')
        return;
    DIR* d = opendir(path.c_str());
    if (!d)
        return;
    struct dirent* dir;
    while ((dir = readdir(d)) != NULL)
    {
        if (dir->d_type != DT_DIR)
            walker(path + dir->d_name);
        else if (strcmp(dir->d_name, ".") != 0 && strcmp(dir->d_name, "..") != 0)
            WalkFiles(path + dir->d_name + '/', walker);
    }
    closedir(d);
}

static bool DecodeImage(
    const std::string& file, std::vector<uint8_t>& output, int& height_, int& width_, int& depths_)
{
    struct jpeg_decompress_struct cinfo;
    FILE* infile;
    JSAMPARRAY buffer;
    int row_stride;

    if ((infile = fopen(file.c_str(), "rb")) == NULL)
        return false;

    struct my_error_mgr
    {
        struct jpeg_error_mgr pub;
        jmp_buf setjmp_buffer;
    };

    my_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = [](j_common_ptr cinfo) -> void {
        my_error_mgr* myerr = (my_error_mgr*)cinfo->err;
        (*cinfo->err->output_message)(cinfo);
        longjmp(myerr->setjmp_buffer, 1);
    };
    if (setjmp(jerr.setjmp_buffer))
    {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return false;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);

    (void)jpeg_read_header(&cinfo, TRUE);
    (void)jpeg_start_decompress(&cinfo);
    height_ = cinfo.output_height;
    width_ = cinfo.output_width;
    depths_ = cinfo.output_components;
    assert(depths_ == 3 || depths_ == 1);

    row_stride = cinfo.output_width * cinfo.output_components;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

    output.resize(height_ * width_ * depths_);
    uint8_t* hptr = output.data();
    while (cinfo.output_scanline < cinfo.output_height)
    {
        (void)jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(hptr, buffer[0], row_stride);
        hptr += row_stride;
    }
    (void)jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return true;
}

template <class T>
static int placeholder_iterate_image(std::string image_dir,
                                     T* image_offset,
                                     const std::vector<int>& shape,
                                     const std::string& image_format = "NHWC",
                                     float rescale = 1.0f / 255)
{
    static std::vector<std::vector<std::string>> dict;
    if (image_dir.size() == 0)
    {
        if (!dict.size())
            return -1;
        int cls = dict.size() - 1;
        auto& it = dict[cls];
        std::string path = it.back();
        it.pop_back();
        if (it.size() == 0)
            dict.pop_back();
        std::vector<uint8_t> output;
        int height_, width_, depths_;
        if (!DecodeImage(path, output, height_, width_, depths_))
            return placeholder_iterate_image("", image_offset, shape);
        // printf("%d => %d %d %d\n", cls, height_, width_, depths_);
        uint8_t* image_ptr = output.data();
        std::vector<int> stride;
        int height, width, depth;
        if (image_format == "NCHW")
            depth = shape[1], height = shape[2], width = shape[3],
            stride = {width, 1, width * height};
        else
            depth = shape[3], height = shape[1], width = shape[2],
            stride = {width * depth, depth, 1};
        assert(depth == 3);
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                for (int d = 0; d < depth; ++d)
                {
                    int ih = h * height_ / height, iw = w * width_ / width;
                    *(image_offset + h * stride[0] + w * stride[1] + d * stride[2]) =
                        *(image_ptr + ih * width_ * depths_ + iw * depths_ +
                          (depths_ == 3 ? d : 0)) *
                        rescale;
                }
            }
        }
        return cls;
    }
    if (image_dir.size() > 0 && image_dir.back() != '/')
        image_dir += '/';
    std::map<std::string, std::vector<std::string>> hash_idx;
    WalkFiles(image_dir, [&](const std::string& file) {
        std::string cls = file.substr(image_dir.size());
        int at = cls.find('/');
        if (at < 0)
            return;
        cls = cls.substr(0, at);
        std::string ext;
        for (int i = file.size() - 1; i >= 0; --i)
            if (file[i] == '.')
            {
                ext = file.substr(i + 1);
                break;
            }
        for (char& c : ext)
            if (islower(c))
                c -= 'a' - 'A';
        if (ext != "JPG")
            return;
        hash_idx[cls].push_back(file);
    });
    dict.clear();
    for (auto& it : hash_idx)
        dict.emplace_back(std::move(it.second));
    return placeholder_iterate_image("", image_offset, shape);
}

int main()
{
    if (NNFUSION_GRAPH_INPUT_NUM != 1 || NNFUSION_GRAPH_OUTPUT_NUM != 1)
        throw std::runtime_error(
            (std::string(
                 "Image Test only supports |inputs nodes| = |inputs nodes| = 1, received: ") +
             std::to_string(NNFUSION_GRAPH_INPUT_NUM) + ", " +
             std::to_string(NNFUSION_GRAPH_OUTPUT_NUM))
                .c_str());

    std::vector<int> shape = NNFUSION_GRAPH_INPUT_SHAPE_0, oshape = NNFUSION_GRAPH_OUTPUT_SHAPE_0;
    if (shape.size() != 4 || oshape.size() != 2 || shape[0] != oshape[0] || shape[3] != 3)
        throw std::runtime_error("Invalid graph input or output to match image inference test.");

    int n_class = oshape[1];
    const char* image_path = getenv("IMAGES");
    if (!image_path)
        throw std::runtime_error(
            "Except to fill image path directory as environment variable: IMAGES");

    size_t numElem = 1;
    for (int i = 0; i < shape.size(); numElem *= shape[i++])
        ;

    cuda_init();

    float *image_input, *d_image_input, *logits_output, *d_logits_output;
    assert(0 == cudaMallocHost(&image_input, sizeof(*image_input) * numElem));
    assert(0 == cudaMalloc((void**)&d_image_input, sizeof(*image_input) * numElem));
    assert(0 == cudaMallocHost(&logits_output, sizeof(*logits_output) * shape[0] * n_class));
    assert(0 == cudaMalloc((void**)&d_logits_output, sizeof(*logits_output) * shape[0] * n_class));

    std::vector<int> sparse_cls(shape[0]);
    const char* load_path = image_path;

    int tot_positive = 0, tot_sample = 0, num_batch = shape[0];
    while (true)
    {
        for (int i = 0; i < shape[0]; ++i)
        {
            int cls = placeholder_iterate_image<float>(
                load_path, image_input + numElem / shape[0] * i, shape);
            if (cls < 0)
            {
                num_batch = i;
                break;
            }
            sparse_cls[i] = cls;
            load_path = "";
        }
        if (!num_batch)
            break;
        assert(0 == cudaMemcpy(d_image_input,
                               image_input,
                               sizeof(*image_input) * numElem,
                               cudaMemcpyHostToDevice));
        kernel_entry(d_image_input, d_logits_output);
        assert(0 == cudaMemcpy(logits_output,
                               d_logits_output,
                               sizeof(*logits_output) * num_batch * n_class,
                               cudaMemcpyDeviceToHost));
        assert(0 == cudaDeviceSynchronize());
        int acc = 0;
        for (int i = 0; i < num_batch; ++i)
        {
            int it = 0;
            for (int j = 1; j < n_class; ++j)
                if (logits_output[i * n_class + it] < logits_output[i * n_class + j])
                    it = j;
            tot_positive += (it == sparse_cls[i]), ++tot_sample;
            acc += (it == sparse_cls[i]);
        }
        printf(">> Batch Accuracy = %.2lf%%;\n", acc * 1e2 / num_batch);
    }
    printf("Total accuracy = %.2lf%%;\n", tot_positive * 1e2 / tot_sample);
    assert(0 == cudaFreeHost(image_input));
    assert(0 == cudaFreeHost(logits_output));
    assert(0 == cudaFree(d_image_input));
    assert(0 == cudaFree(d_logits_output));
    return 0;
}
