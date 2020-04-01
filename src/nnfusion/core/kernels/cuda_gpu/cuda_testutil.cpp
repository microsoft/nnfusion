// Microsoft (c) 2019, Wenxiang
#include "cuda_testutil.hpp"
using namespace nnfusion;
using namespace nnfusion::kernels;

void cuda::test_cudaMemcpyDtoH(CodeWriter& writer, shared_ptr<nnfusion::descriptor::Tensor> tensor)
{
    writer << "cudaMemcpy(" << tensor->get_name() << "_host, " << tensor->get_name() << ", "
           << tensor->size() << ", "
           << "cudaMemcpyDeviceToHost);\n";
}

void cuda::test_cudaMemcpyHtoD(CodeWriter& writer, shared_ptr<nnfusion::descriptor::Tensor> tensor)
{
    writer << "cudaMemcpy(" << tensor->get_name() << ", " << tensor->get_name() << "_host, "
           << tensor->size() << ", "
           << "cudaMemcpyHostToDevice);\n";
}

void cuda::test_cudaMalloc(CodeWriter& writer, shared_ptr<nnfusion::descriptor::Tensor> tensor)
{
    writer << tensor->get_element_type().c_type_string() << "* " << tensor->get_name() << ";\n"
           << "cudaMalloc((void**)&" << tensor->get_name() << "," << tensor->size() << ");\n";
}

vector<float> cuda::test_hostData(CodeWriter& writer,
                                  shared_ptr<nnfusion::descriptor::Tensor> tensor)
{
    size_t size = tensor->size(false);
    vector<float> d;
    vector<string> dstr;
    float sign = 1;
    for (int i = 0; i < size; i++)
    {
        d.push_back(sign * ((rand() / double(RAND_MAX)) * 256.0 - 512.0));
        dstr.push_back(to_string(d.back()));
        sign *= -1;
    }

    writer << tensor->get_element_type().c_type_string() << " " << tensor->get_name() << "_host[] ="
           << "{" << join(dstr, ", ") << "};\n";
    return d;
}

vector<float> cuda::test_hostData(CodeWriter& writer,
                                  shared_ptr<nnfusion::descriptor::Tensor> tensor,
                                  vector<float>& d)
{
    size_t size = tensor->size(false);
    vector<string> dstr;
    for (int i = 0; i < size; i++)
    {
        dstr.push_back(to_string(d[i]));
    }

    writer << tensor->get_element_type().c_type_string() << " " << tensor->get_name()
           << "_host_result[] ="
           << "{" << join(dstr, ", ") << "};\n"
           << tensor->get_element_type().c_type_string() << " " << tensor->get_name() << "_host["
           << size << "];\n";
    return d;
}

void cuda::test_compare(CodeWriter& writer, shared_ptr<nnfusion::descriptor::Tensor> tensor)
{
    size_t size = tensor->size(false);
    writer << "for(int i = 0; i < " << size << "; i++)\n"
           << "{\n"
           << "    if(abs(" << tensor->get_name() << "_host_result[i] - " << tensor->get_name()
           << "_host[i]) > 0.00005)\n"
           << "    {\n"
           << "        printf(\"Error on tensor:" << tensor->get_name() << "\");\n"
           << "        return false;\n"
           << "    }\n"
           << "};\n";
}
