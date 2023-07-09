// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "lstm.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::LstmEigen::LstmEigen(shared_ptr<KernelContext> ctx)
    : EigenKernelEmitter(ctx)
{
    auto input_shape_0 = ctx->inputs[0]->get_shape();
    SEQ_LEN = input_shape_0[0];
    SIZE_BATCH = input_shape_0[1];
    SIZE_INPUT = input_shape_0[2];
    auto input_shape_1 = ctx->inputs[1]->get_shape();
    NUM_DRT = input_shape_1[0];
    auto input_shape_2 = ctx->inputs[2]->get_shape();
    SIZE_HIDDEN = input_shape_2[2];
    // attr
    auto generic_op = static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr());
    direction = generic_op->localOpConfig.getRoot()["direction"];
    // hidden_size = generic_op->localOpConfig.getRoot()["hidden_size"];
    input_forget = generic_op->localOpConfig.getRoot()["input_forget"];
    // gates
    NUM_GATE = 4;
    SIZE_GATE = SIZE_BATCH * SIZE_HIDDEN;

    std::stringstream tag;
    tag << "Eigen_lstm"
        << "_i_" << join(input_shape_0, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cpu::LstmEigen::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    // malloc
    auto malloc_code = nnfusion::op::create_code_from_template(
        R"(float* sumForward = (float*) malloc(sizeof(float) * @SIZE_SUM@);
float* sumBackward = (float*) malloc(sizeof(float) * @SIZE_SUM@);
float* gateI = (float*) malloc(sizeof(float) * @SIZE_GATE@); Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> TgateI(gateI, {@SIZE_BATCH@, @SIZE_HIDDEN@});
float* gateO = (float*) malloc(sizeof(float) * @SIZE_GATE@); Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> TgateO(gateO, {@SIZE_BATCH@, @SIZE_HIDDEN@});
float* gateF = (float*) malloc(sizeof(float) * @SIZE_GATE@); Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> TgateF(gateF, {@SIZE_BATCH@, @SIZE_HIDDEN@});
float* gateC = (float*) malloc(sizeof(float) * @SIZE_GATE@); Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> TgateC(gateC, {@SIZE_BATCH@, @SIZE_HIDDEN@});
float* currH = (float*) malloc(sizeof(float) * @SIZE_GATE@);
float* currC = (float*) malloc(sizeof(float) * @SIZE_GATE@);
float* sumProj, *currX, *currW, *currR, *currB, *currO;
)",
        {{"SIZE_GATE", SIZE_GATE},
         {"SIZE_SUM", SEQ_LEN * NUM_GATE * SIZE_GATE},
         {"SIZE_BATCH", SIZE_BATCH},
         {"SIZE_HIDDEN", SIZE_HIDDEN}});
    lu << malloc_code;

    // compute input
    lu << "const int min_cost_per_shard = 10000;\n";
    lu << "int SEQ_LEN = " << SEQ_LEN << ";\n";
    lu << "int num_shards = std::max(1, std::min(thread_pool->NumThreads(), SEQ_LEN / "
          "min_cost_per_shard));\n";
    lu << "int block_size = (SEQ_LEN + num_shards - 1) / num_shards;\n";
    lu << "if (block_size > SEQ_LEN)"
       << " num_shards = 1;\n";
    lu << "auto func = [&](int __rank__)\n{\n";
    lu << "int start = block_size * __rank__;\n";
    lu << "int end = std::min(block_size * (__rank__ + 1), SEQ_LEN);\n";

    lu << "for (int seqI = start; seqI < end; seqI++)";
    lu.block_begin();
    if (direction == "bidirectional")
    {
        lu << "currX = input0 + seqI * " << SIZE_BATCH * SIZE_INPUT << ";\n";
        lu << "currW = input1;\n"
           << "currB = input3;\n";
        lu << "sumProj = sumForward + seqI * " << NUM_GATE * SIZE_GATE << ";\n";
        emit_compute_input_helper(lu);
        lu << "currW = input1 + " << NUM_GATE * SIZE_HIDDEN * SIZE_INPUT << ";\n";
        lu << "currB = input3 + " << NUM_DRT * NUM_GATE * SIZE_HIDDEN << ";\n";
        lu << "sumProj = sumBackward + (" << SEQ_LEN << " - 1 - seqI) * " << NUM_GATE * SIZE_GATE
           << ";\n";
        emit_compute_input_helper(lu);
    }
    else
    {
        return nullptr;
    }
    lu.block_end();
    lu << "};\n";
    lu << "thread_pool->ParallelFor(num_shards, func);\n\n";

    // compute hidden
    auto compute_hidden_forward_code = nnfusion::op::create_code_from_template(
        R"(sumProj = sumForward;
currR = input2;
memcpy(currH, input4, sizeof(float) * @SIZE_GATE@);
memcpy(currC, input5, sizeof(float) * @SIZE_GATE@);
currO = output0;
for (int seqI = 0; seqI < @SEQ_LEN@; seqI++)
)",
        {{"SEQ_LEN", SEQ_LEN}, {"SIZE_GATE", SIZE_GATE}});
    auto compute_hidden_backward_code = nnfusion::op::create_code_from_template(
        R"(sumProj = sumBackward;
currR = input2 + @SIZE_R@;
memcpy(currH, input4 + @SIZE_GATE@, sizeof(float) * @SIZE_GATE@);
memcpy(currC, input5 + @SIZE_GATE@, sizeof(float) * @SIZE_GATE@);
currO = output0 + @SIZE_GATE@;
for (int seqI = @SEQ_LEN@; seqI > -1; seqI--)
)",
        {{"SIZE_R", NUM_GATE * SIZE_HIDDEN * SIZE_HIDDEN},
         {"SIZE_GATE", SIZE_GATE},
         {"SEQ_LEN", SEQ_LEN - 1}});

    if (direction == "bidirectional")
    {
        lu << compute_hidden_forward_code;
        lu.block_begin();
        emit_compute_hidden_helper(lu);
        lu.block_end();
        lu << compute_hidden_backward_code;
        lu.block_begin();
        emit_compute_hidden_helper(lu);
        lu.block_end();
    }
    else
    {
        return nullptr;
    }

    // free
    lu << "free(gateI); free(gateO); free(gateF); free(gateC); free(currH); free(currC); "
          "free(sumForward); free(sumBackward);";

    return _lu;
}

LanguageUnit_p cpu::LstmEigen::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::eigen_tensor);
    _lu->require(header::eigen_spatial_convolution);
    _lu->require(header::mlas);
    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Lstm",                                                                    // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("eigen").Priority(4), // attrs
    cpu::LstmEigen)

void cpu::LstmEigen::emit_compute_input_helper(nnfusion::codegen::CodeWriter& lu)
{
    auto code = nnfusion::op::create_code_from_template(
        R"(MlasGemm(CblasNoTrans, CblasTrans, @SIZE_BATCH@, @HIDDEN_4@, @SIZE_INPUT@, 1.0f, currX, @SIZE_INPUT@, currW, @SIZE_INPUT@, 0.0f, sumProj, @HIDDEN_4@, thread_pool);
for (int iB = 0; iB < @SIZE_BATCH@; iB++) {
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> TB1(currB, {@HIDDEN_4@});
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> TB2(currB + @HIDDEN_4@, {@HIDDEN_4@});
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> TsumProj(sumProj + iB * @HIDDEN_4@, {@HIDDEN_4@});
    TsumProj.device(*(thread_pool->GetDevice())) = TsumProj + TB1 + TB2; 
}
)",
        {{"NUM_GATE", NUM_GATE},
         {"SIZE_BATCH", SIZE_BATCH},
         {"SIZE_INPUT", SIZE_INPUT},
         {"SIZE_HIDDEN", SIZE_HIDDEN},
         {"HIDDEN_4", NUM_GATE * SIZE_HIDDEN}});

    lu << code;
}

void cpu::LstmEigen::emit_compute_hidden_helper(nnfusion::codegen::CodeWriter& lu)
{
    auto code = nnfusion::op::create_code_from_template(
        R"(MlasGemm(CblasNoTrans, CblasTrans, @SIZE_BATCH@, @HIDDEN_4@, @SIZE_HIDDEN@, 1.0f, currH, @SIZE_HIDDEN@, currR, @SIZE_HIDDEN@, 1.0f, sumProj, @HIDDEN_4@, thread_pool);
for (int iB = 0; iB < @SIZE_BATCH@; iB++) {
    memcpy(gateI + iB * @SIZE_HIDDEN@, sumProj + iB * @HIDDEN_4@ + 0 * @SIZE_HIDDEN@, sizeof(float) * @SIZE_HIDDEN@);
    memcpy(gateO + iB * @SIZE_HIDDEN@, sumProj + iB * @HIDDEN_4@ + 1 * @SIZE_HIDDEN@, sizeof(float) * @SIZE_HIDDEN@);
    memcpy(gateF + iB * @SIZE_HIDDEN@, sumProj + iB * @HIDDEN_4@ + 2 * @SIZE_HIDDEN@, sizeof(float) * @SIZE_HIDDEN@);
    memcpy(gateC + iB * @SIZE_HIDDEN@, sumProj + iB * @HIDDEN_4@ + 3 * @SIZE_HIDDEN@, sizeof(float) * @SIZE_HIDDEN@);            
}
TgateI.device(*(thread_pool->GetDevice())) = TgateI.sigmoid();
TgateO.device(*(thread_pool->GetDevice())) = TgateO.sigmoid();
TgateF.device(*(thread_pool->GetDevice())) = TgateF.sigmoid();
TgateC.device(*(thread_pool->GetDevice())) = TgateC.tanh();
Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> TcurrC(currC, {@SIZE_BATCH@, @SIZE_HIDDEN@});
Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> TcurrH(currH, {@SIZE_BATCH@, @SIZE_HIDDEN@});
TcurrC.device(*(thread_pool->GetDevice())) = TgateF * TcurrC + TgateI * TgateC;
TcurrH.device(*(thread_pool->GetDevice())) = TgateO * TcurrC.tanh();
memcpy(currO + seqI * @SIZE_OUT@, currH, sizeof(float) * @SIZE_GATE@);
sumProj = sumProj + @ONE_BATCH_SIZE_SUM@;
)",
        {{"NUM_GATE", NUM_GATE},
         {"SIZE_BATCH", SIZE_BATCH},
         {"SIZE_INPUT", SIZE_INPUT},
         {"SIZE_HIDDEN", SIZE_HIDDEN},
         {"SIZE_GATE", SIZE_GATE},
         {"HIDDEN_4", NUM_GATE * SIZE_HIDDEN},
         {"SIZE_OUT", NUM_DRT * SIZE_GATE},
         {"ONE_BATCH_SIZE_SUM", NUM_GATE * SIZE_BATCH * SIZE_HIDDEN}});

    lu << code;
}