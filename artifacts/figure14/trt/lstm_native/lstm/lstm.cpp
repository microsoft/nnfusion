#include "NvInfer.h"
#include "NvUtils.h"
#include "cuda_runtime_api.h"
#include "argsParser.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cstdio>
#include "logger.h"
#include "common.h"
#include <cuda_profiler_api.h>

#define NUM_RUNS 3

namespace
{ // anonymous
// Logger gLogger;

samplesCommon::Args gArgs;
const std::string gSampleName = "TensorRT.sample_lstm";

// The model used by this sample was trained using github repository:
// https://github.com/crazydonkey200/tensorflow-char-rnn
//
// The data set used: tensorflow-char-rnn/data/tiny_shakespeare.txt
//
// The command used to train:
// python train.py --data_file=data/tiny_shakespeare.txt --num_epochs=100 --num_layer=2 --hidden_size=512 --embedding_size=512 --dropout=.5
//
// Epochs trained: 100
// Test perplexity: 4.940
//
// Layer0 and Layer1 weights matrices are added as RNNW_L0_NAME and RNNW_L1_NAME, respectively.
// Layer0 and Layer1 bias are added as RNNB_L0_NAME and RNNB_L1_NAME, respectively.
// Embedded is added as EMBED_NAME.
// fc_w is added as FCW_NAME.
// fc_b is added as FCB_NAME.

// // A mapping from character to index used by the tensorflow model
// static std::map<char, int> char_to_id{{'\n', 0}, {'!', 1}, {' ', 2}, {'$', 3}, {'\'', 4}, {'&', 5}, {'-', 6}, {',', 7}, {'.', 8}, {'3', 9}, {';', 10}, {':', 11}, {'?', 12}, {'A', 13}, {'C', 14}, {'B', 15}, {'E', 16}, {'D', 17}, {'G', 18}, {'F', 19}, {'I', 20}, {'H', 21}, {'K', 22}, {'J', 23}, {'M', 24}, {'L', 25}, {'O', 26}, {'N', 27}, {'Q', 28}, {'P', 29}, {'S', 30}, {'R', 31}, {'U', 32}, {'T', 33}, {'W', 34}, {'V', 35}, {'Y', 36}, {'X', 37}, {'Z', 38}, {'a', 39}, {'c', 40}, {'b', 41}, {'e', 42}, {'d', 43}, {'g', 44}, {'f', 45}, {'i', 46}, {'h', 47}, {'k', 48}, {'j', 49}, {'m', 50}, {'l', 51}, {'o', 52}, {'n', 53}, {'q', 54}, {'p', 55}, {'s', 56}, {'r', 57}, {'u', 58}, {'t', 59}, {'w', 60}, {'v', 61}, {'y', 62}, {'x', 63}, {'z', 64}};

// // A mapping from index to character used by the tensorflow model.
// static std::vector<char> id_to_char{{'\n', '!', ' ', '$', '\'', '&', '-', ',',
//                                      '.', '3', ';', ':', '?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K',
//                                      'J', 'M', 'L', 'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X',
//                                      'Z', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o',
//                                      'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z'}};

// Information describing the network
const int LAYER_COUNT = 10;
const int BATCH_SIZE = 1;
const int HIDDEN_SIZE = 256;
const int SEQ_SIZE = 100;
const int DATA_SIZE = HIDDEN_SIZE;

const int NUM_BINDINGS = 7;
const int INPUT_IDX = 0;
const int HIDDEN_IN_IDX = 1;
const int CELL_IN_IDX = 2;
const int HIDDEN_OUT_IDX = 3;
const int CELL_OUT_IDX = 4;
const int OUTPUT_IDX = 5;
const int SEQ_LEN_IN_IDX = 6;

const char *INPUT_BLOB_NAME = "data";
const char *HIDDEN_IN_BLOB_NAME = "hiddenIn";
const char *CELL_IN_BLOB_NAME = "cellIn";
const char *HIDDEN_OUT_BLOB_NAME = "hiddenOut";
const char *CELL_OUT_BLOB_NAME = "cellOut";
const char *OUTPUT_BLOB_NAME = "pred";
const char *SEQ_LEN_IN_BLOB_NAME = "seqLen";

const char *gNames[NUM_BINDINGS] = {
    INPUT_BLOB_NAME,
    HIDDEN_IN_BLOB_NAME,
    CELL_IN_BLOB_NAME,
    HIDDEN_OUT_BLOB_NAME,
    CELL_OUT_BLOB_NAME,
    OUTPUT_BLOB_NAME,
    SEQ_LEN_IN_BLOB_NAME};

const int gSizes[NUM_BINDINGS] = {
    BATCH_SIZE * SEQ_SIZE * DATA_SIZE,
    LAYER_COUNT *BATCH_SIZE *HIDDEN_SIZE,
    LAYER_COUNT *BATCH_SIZE *HIDDEN_SIZE,
    LAYER_COUNT *BATCH_SIZE *HIDDEN_SIZE,
    LAYER_COUNT *BATCH_SIZE *HIDDEN_SIZE,
    BATCH_SIZE *SEQ_SIZE *HIDDEN_SIZE,
    BATCH_SIZE};

const bool gHostInput[NUM_BINDINGS] = {
    false, false, false, false, false, false, true};

// const string RNNW_L0_NAME = "rnn_multi_rnn_cell_cell_0_basic_lstm_cell_kernel";
// const string RNNB_L0_NAME = "rnn_multi_rnn_cell_cell_0_basic_lstm_cell_bias";
// const string RNNW_L1_NAME = "rnn_multi_rnn_cell_cell_1_basic_lstm_cell_kernel";
// const string RNNB_L1_NAME = "rnn_multi_rnn_cell_cell_1_basic_lstm_cell_bias";
// const string FCW_NAME = "softmax_softmax_w";
// const string FCB_NAME = "softmax_softmax_b";
// const string EMBED_NAME = "embedding";

// unordered_set<string> weightNames{{RNNW_L0_NAME, RNNB_L0_NAME, RNNW_L1_NAME,
//                                    RNNB_L1_NAME, FCW_NAME, FCB_NAME, EMBED_NAME}};

std::vector<Weights> weight_storage;

} // anonymous namespace

using namespace nvinfer1;

//!
//! \brief Load requested weights from a formatted file into a map.
//!
//! \param file Path to weights file. File has to be the formatted dump from
//!        the dumpTFWts.py script. Otherwise, this function will not work as
//!        intended.
//! \param names The names for all of the weights the user wishes to extract.
//!
//! \return A map containing the extracted weights.
//!
//! \note  Weight V2 files are in a very simple space delimited format.
//!        <number of buffers>
//!        for each buffer: [name] [type] [shape] <data as binary blob>\n
//!        Note: type is the integer value of the DataType enum in NvInfer.h.
//!
// std::map<std::string, Weights> loadWeights(const std::string file, std::unordered_set<std::string> names)
// {
//     std::map<std::string, Weights> weightMap;

//     // std::ifstream input(file, std::ios_base::binary);
//     // assert(input.is_open() && "Unable to load weight file.");

//     // int32_t count;
//     // input >> count;
//     // assert(count > 0 && "Invalid weight map file.");

//     size_t count_lstm_kernel = 8 * HIDDEN_SIZE * HIDDEN_SIZE;
//     size_t count_lstm_bias = 4 * HIDDEN_SIZE;

//     {
//         std::string name = RNNW_L0_NAME;
//         Weights wt{DataType::kFLOAT, nullptr, 0};
//         char *wtVals = static_cast<char *>(malloc(sizeof(float) * count_lstm_kernel));
//         wt.values = wtVals;
//         wt.count = count_lstm_kernel;
//         weightMap[name] = wt;
//     }
//     {
//         std::string name = RNNW_L1_NAME;
//         Weights wt{DataType::kFLOAT, nullptr, 0};
//         char *wtVals = static_cast<char *>(malloc(sizeof(float) * count_lstm_kernel));
//         wt.values = wtVals;
//         wt.count = count_lstm_kernel;
//         weightMap[name] = wt;
//     }
//     {
//         std::string name = RNNB_L0_NAME;
//         Weights wt{DataType::kFLOAT, nullptr, 0};
//         char *wtVals = static_cast<char *>(malloc(sizeof(float) * count_lstm_bias));
//         wt.values = wtVals;
//         wt.count = count_lstm_bias;
//         weightMap[name] = wt;
//     }
//     {
//         std::string name = RNNB_L1_NAME;
//         Weights wt{DataType::kFLOAT, nullptr, 0};
//         char *wtVals = static_cast<char *>(malloc(sizeof(float) * count_lstm_bias));
//         wt.values = wtVals;
//         wt.count = count_lstm_bias;
//         weightMap[name] = wt;
//     }

//     // while (count--)
//     // {
//     //     if (names.empty())
//     //         break;

//     //     Weights wt{DataType::kFLOAT, nullptr, 0};

//     //     // parse name and DataType
//     //     std::string name;
//     //     uint32_t type;
//     //     input >> name >> std::dec >> type;
//     //     wt.type = static_cast<DataType>(type);

//     //     // extract shape
//     //     std::string temp, shape;
//     //     std::getline(std::getline(input, temp, '('), shape, ')');

//     //     // calculate count based on shape
//     //     wt.count = 1;
//     //     std::istringstream shapeStream(shape);
//     //     while (std::getline(shapeStream, temp, ','))
//     //         wt.count *= std::stoul(temp);
//     //     size_t numOfBytes = samplesCommon::getElementSize(wt.type) * wt.count;

//     //     // skip reading of weights if name is not in the set of names requested for extraction
//     //     if (names.find(name) == names.end())
//     //     {
//     //         input.seekg(input.tellg() + static_cast<std::streamoff>(2 + numOfBytes));
//     //         continue;
//     //     }
//     //     else
//     //     {
//     //         names.erase(name);
//     //     }

//     //     // Read weight values
//     //     input.seekg(input.tellg() + static_cast<std::streamoff>(1)); // skip space char
//     //     char* wtVals = static_cast<char*>(malloc(numOfBytes));
//     //     input.read(wtVals, numOfBytes);
//     //     input.seekg(input.tellg() + static_cast<std::streamoff>(1)); // skip new-line char
//     //     wt.values = wtVals;

//     //     weightMap[name] = wt;
//     // }

//     // input.close();

//     return weightMap;
// }

//!
//! \brief Converts RNN weights from TensorFlow's format to TensorRT's format.
//!
//! \param input Weights that are stored in TensorFlow's format.
//!
//! \return Converted weights in TensorRT's format.
//!
//! \note TensorFlow weight parameters for BasicLSTMCell are formatted as:
//!       Each [WR][icfo] is hiddenSize sequential elements.
//!       CellN  Row 0: WiT, WcT, WfT, WoT
//!       CellN  Row 1: WiT, WcT, WfT, WoT
//!       ...
//!       CellN RowM-1: WiT, WcT, WfT, WoT
//!       CellN RowM+0: RiT, RcT, RfT, RoT
//!       CellN RowM+1: RiT, RcT, RfT, RoT
//!       ...
//!       CellNRow2M-1: RiT, RcT, RfT, RoT
//!
//!       TensorRT expects the format to laid out in memory:
//!       CellN: Wi, Wc, Wf, Wo, Ri, Rc, Rf, Ro
//!
Weights convertRNNWeights(Weights input)
{
    float *ptr = static_cast<float *>(malloc(sizeof(float) * input.count));
    int dims[4]{2, HIDDEN_SIZE, 4, HIDDEN_SIZE};
    int order[4]{0, 3, 1, 2};
    utils::reshapeWeights(input, dims, order, ptr, 4);
    utils::transposeSubBuffers(ptr, DataType::kFLOAT, 2, HIDDEN_SIZE * HIDDEN_SIZE, 4);
    return Weights{input.type, ptr, input.count};
}

//!
//! \brief Converts RNN Biases from TensorFlow's format to TensorRT's format.
//!
//! \param input Biases that are stored in TensorFlow's format.
//!
//! \return Converted bias in TensorRT's format.
//!
//! \note TensorFlow bias parameters for BasicLSTMCell are formatted as:
//!       CellN: Bi, Bc, Bf, Bo
//!
//!       TensorRT expects the format to be:
//!       CellN: Wi, Wc, Wf, Wo, Ri, Rc, Rf, Ro
//!
//!       Since tensorflow already combines U and W,
//!       we double the size and set all of U to zero.
Weights convertRNNBias(Weights input)
{
    const int sizeOfElement = samplesCommon::getElementSize(input.type);
    char *ptr = static_cast<char *>(malloc(sizeOfElement * input.count * 2));
    const char *iptr = static_cast<const char *>(input.values);
    std::copy(iptr, iptr + 4 * HIDDEN_SIZE * sizeOfElement, ptr);
    std::fill(ptr + sizeOfElement * input.count, ptr + sizeOfElement * input.count * 2, 0);
    return Weights{input.type, ptr, input.count * 2};
}

//!
//! \brief Add to the network and configure the RNNv2 layer.
//!
//! \param network The network that will be used to build the engine.
//! \param weightMap Map that contains all the weights required by the model.
//!
//! \return Configured and added RNNv2 layer.
//!
IRNNv2Layer *addRNNv2Layer(INetworkDefinition *network, std::map<std::string, Weights> &weightMap)
{
    // Initialize data, hiddenIn, cellIn, and seqLenIn inputs into RNN Layer
    auto data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims2(SEQ_SIZE, DATA_SIZE));
    assert(data != nullptr);

    auto hiddenIn = network->addInput(HIDDEN_IN_BLOB_NAME, DataType::kFLOAT, Dims2(LAYER_COUNT, HIDDEN_SIZE));
    assert(hiddenIn != nullptr);

    auto cellIn = network->addInput(CELL_IN_BLOB_NAME, DataType::kFLOAT, Dims2(LAYER_COUNT, HIDDEN_SIZE));
    assert(cellIn != nullptr);

    auto seqLenIn = network->addInput(SEQ_LEN_IN_BLOB_NAME, DataType::kINT32, Dims{});
    assert(seqLenIn != nullptr);

    // create an RNN layer w/ 2 layers and 512 hidden states
    auto rnn = network->addRNNv2(*data, LAYER_COUNT, HIDDEN_SIZE, SEQ_SIZE, RNNOperation::kLSTM);
    assert(rnn != nullptr);

    // Set RNNv2 optional inputs
    std::cout << "num of outputs: " << rnn->getNbOutputs() << std::endl;
    rnn->getOutput(0)->setName("RNN output");
    rnn->setHiddenState(*hiddenIn);
    if (rnn->getOperation() == RNNOperation::kLSTM)
        rnn->setCellState(*cellIn);

    // Specify sequence lengths.  Note this can be omitted since we are always using the maximum
    // sequence length, but for illustrative purposes we explicitly pass in sequence length data
    // in the sample
    rnn->setSequenceLengths(*seqLenIn);
    // std::cout<<"seqLen: "<<
    // Give sequence lengths directly from host
    seqLenIn->setLocation(TensorLocation::kHOST);

    size_t count_lstm_kernel = 8 * HIDDEN_SIZE * HIDDEN_SIZE;
    Weights wt_kernel{DataType::kFLOAT, nullptr, 0};
    wt_kernel.values = static_cast<char *>(malloc(sizeof(float) * count_lstm_kernel));
    wt_kernel.count = count_lstm_kernel;

    size_t count_lstm_bias = 4 * HIDDEN_SIZE;
    Weights wt_bias{DataType::kFLOAT, nullptr, 0};
    wt_bias.values = static_cast<char *>(malloc(sizeof(float) * count_lstm_bias));
    wt_bias.count = count_lstm_bias;

    std::vector<nvinfer1::RNNGateType> gateOrder({nvinfer1::RNNGateType::kINPUT,
                                                  nvinfer1::RNNGateType::kCELL,
                                                  nvinfer1::RNNGateType::kFORGET,
                                                  nvinfer1::RNNGateType::kOUTPUT});

    const nvinfer1::DataType dataType = static_cast<nvinfer1::DataType>(wt_kernel.type);

    for (int idx_layer = 0; idx_layer < LAYER_COUNT; idx_layer++)
    {
        Weights rnnw = convertRNNWeights(wt_kernel);
        Weights rnnb = convertRNNBias(wt_bias);

        const float *wts = static_cast<const float *>(rnnw.values);
        const float *biases = static_cast<const float *>(rnnb.values);

        size_t kernelOffset = 0, biasOffset = 0;
        for (int gateIndex = 0, numGates = gateOrder.size(); gateIndex < 2 * numGates; gateIndex++)
        {
            // extract weights and bias for a given gate and layer
            Weights gateWeight{dataType, wts + kernelOffset, DATA_SIZE * HIDDEN_SIZE};
            Weights gateBias{dataType, biases + biasOffset, HIDDEN_SIZE};

            // set weights and bias for given gate
            rnn->setWeightsForGate(idx_layer, gateOrder[gateIndex % numGates], (gateIndex < numGates), gateWeight);
            rnn->setBiasForGate(idx_layer, gateOrder[gateIndex % numGates], (gateIndex < numGates), gateBias);

            // Update offsets
            kernelOffset = kernelOffset + DATA_SIZE * HIDDEN_SIZE;
            biasOffset = biasOffset + HIDDEN_SIZE;
        }

        weight_storage.push_back(rnnw);
        weight_storage.push_back(rnnb);
    }

    // // convert tensorflow weight format to trt weight format
    // Weights rnnwL0 = convertRNNWeights(weightMap[RNNW_L0_NAME]);
    // Weights rnnbL0 = convertRNNBias(weightMap[RNNB_L0_NAME]);
    // // Weights rnnwL1 = convertRNNWeights(weightMap[RNNW_L1_NAME]);
    // // Weights rnnbL1 = convertRNNBias(weightMap[RNNB_L1_NAME]);

    // std::vector<nvinfer1::RNNGateType> gateOrder({nvinfer1::RNNGateType::kINPUT,
    //                                               nvinfer1::RNNGateType::kCELL,
    //                                               nvinfer1::RNNGateType::kFORGET,
    //                                               nvinfer1::RNNGateType::kOUTPUT});
    // const nvinfer1::DataType dataType = static_cast<nvinfer1::DataType>(rnnwL0.type);
    // const float *wtsL0 = static_cast<const float *>(rnnwL0.values);
    // const float *biasesL0 = static_cast<const float *>(rnnbL0.values);
    // // const float *wtsL1 = static_cast<const float *>(rnnwL1.values);
    // // const float *biasesL1 = static_cast<const float *>(rnnbL1.values);
    // size_t kernelOffset = 0, biasOffset = 0;
    // for (int gateIndex = 0, numGates = gateOrder.size(); gateIndex < 2 * numGates; gateIndex++)
    // {
    //     // extract weights and bias for a given gate and layer
    //     Weights gateWeightL0{dataType, wtsL0 + kernelOffset, DATA_SIZE * HIDDEN_SIZE};
    //     Weights gateBiasL0{dataType, biasesL0 + biasOffset, HIDDEN_SIZE};
    //     // Weights gateWeightL1{dataType, wtsL1 + kernelOffset, DATA_SIZE * HIDDEN_SIZE};
    //     // Weights gateBiasL1{dataType, biasesL1 + biasOffset, HIDDEN_SIZE};

    //     // set weights and bias for given gate
    //     rnn->setWeightsForGate(0, gateOrder[gateIndex % numGates], (gateIndex < numGates), gateWeightL0);
    //     rnn->setBiasForGate(0, gateOrder[gateIndex % numGates], (gateIndex < numGates), gateBiasL0);
    //     // rnn->setWeightsForGate(1, gateOrder[gateIndex % numGates], (gateIndex < numGates), gateWeightL1);
    //     // rnn->setBiasForGate(1, gateOrder[gateIndex % numGates], (gateIndex < numGates), gateBiasL1);

    //     // Update offsets
    //     kernelOffset = kernelOffset + DATA_SIZE * HIDDEN_SIZE;
    //     biasOffset = biasOffset + HIDDEN_SIZE;
    // }

    // // Store the transformed weights in the weight map so the memory can be properly released later.
    // weightMap["rnnwL0"] = rnnwL0;
    // weightMap["rnnbL0"] = rnnbL0;
    // // weightMap["rnnwL1"] = rnnwL1;
    // // weightMap["rnnbL1"] = rnnbL1;

    return rnn;
}

//!
//! \brief Create model using the TensorRT API and build the engine.
//!
//! \param weightMap Map that contains all the weights required by the model.
//! \param modelStream The stream within which the engine is serialized once built.
//!
void APIToModel(std::map<std::string, Weights> &weightMap, IHostMemory **modelStream)
{
    // create the builder
    IBuilder *builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    // create the model to populate the network, then set the outputs and create an engine
    INetworkDefinition *network = builder->createNetwork();

    // add RNNv2 layer and set its parameters
    auto rnn = addRNNv2Layer(network, weightMap);

    // // Transpose FC weights since TensorFlow's weights are transposed when compared to TensorRT
    // utils::transposeSubBuffers((void*) weightMap[FCW_NAME].values, DataType::kFLOAT, 1, HIDDEN_SIZE, VOCAB_SIZE);

    // // add Constant layers for fully connected weights
    // auto fcwts = network->addConstant(Dims2(VOCAB_SIZE, HIDDEN_SIZE), weightMap[FCW_NAME]);

    // // Add matrix multiplication layer for multiplying rnn output with FC weights
    // auto matrixMultLayer = network->addMatrixMultiply(*fcwts->getOutput(0), false, *rnn->getOutput(0), true);
    // assert(matrixMultLayer != nullptr);
    // matrixMultLayer->getOutput(0)->setName("Matrix Multiplicaton output");

    // // Add elementwise layer for adding bias
    // auto fcbias = network->addConstant(Dims2(VOCAB_SIZE, 1), weightMap[FCB_NAME]);
    // auto addBiasLayer = network->addElementWise(*matrixMultLayer->getOutput(0), *fcbias->getOutput(0), ElementWiseOperation::kSUM);
    // assert(addBiasLayer != nullptr);
    // addBiasLayer->getOutput(0)->setName("Add Bias output");

    // // Add TopK layer to determine which character has highest probability.
    // int reduceAxis = 0x1; // reduce across vocab axis
    // auto pred = network->addTopK(*addBiasLayer->getOutput(0), nvinfer1::TopKOperation::kMAX, 1, reduceAxis);
    // assert(pred != nullptr);
    // pred->getOutput(1)->setName(OUTPUT_BLOB_NAME);

    rnn->getOutput(0)->setName(OUTPUT_BLOB_NAME);

    // Mark the outputs for the network
    network->markOutput(*rnn->getOutput(0));
    // pred->getOutput(1)->setType(DataType::kINT32);
    rnn->getOutput(1)->setName(HIDDEN_OUT_BLOB_NAME);
    network->markOutput(*rnn->getOutput(1));
    if (rnn->getOperation() == RNNOperation::kLSTM)
    {
        rnn->getOutput(2)->setName(CELL_OUT_BLOB_NAME);
        network->markOutput(*rnn->getOutput(2));
    }

    // Build the engine
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 25);
    // builder->setFp16Mode(true);
    builder->setFp16Mode(false);
    samplesCommon::enableDLA(builder, gArgs.useDLACore);

    auto engine = builder->buildCudaEngine(*network);
    assert(engine != nullptr);

    // serialize engine and clean up resources
    network->destroy();
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
}

//!
//! \brief Allocate all of the memory required for the input and output buffers.
//!
//! \param engine The engine used at run-time.
//! \param buffers The engine buffers that will be used for input and output.
//! \param data The CPU buffers used to copy back and forth data from the engine.
//! \param indices The indices that the engine has bound specific blobs to.
//!
void allocateMemory(const ICudaEngine &engine, void **buffers, float **data, int *indices)
{
    for (int x = 0; x < NUM_BINDINGS; ++x)
    {
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        indices[x] = engine.getBindingIndex(gNames[x]);
        assert(indices[x] < NUM_BINDINGS);
        assert(indices[x] >= 0);

        // create GPU and CPU buffers and a stream
        data[x] = new float[gSizes[x]];
        std::fill(data[x], data[x] + gSizes[x], 0);
        if (gHostInput[x])
            buffers[indices[x]] = data[x];
        else
            CHECK(cudaMalloc(&buffers[indices[x]], gSizes[x] * sizeof(float)));
    }
}

//!
//! \brief Runs one iteration of the model.
//!
//! \param data The CPU buffers used to copy back and forth data from the engine.
//! \param buffers The engine buffers that will be used for input and output.
//! \param indices The indices that the engine has bound specific blobs to.
//! \param stream The cuda stream used during execution.
//! \param context The TensorRT context used to run the model.
//!
void stepOnce(float **data, void **buffers, int *indices, cudaStream_t &stream, IExecutionContext &context)
{
    CHECK(cudaProfilerStop());
    // DMA the input to the GPU
    // CHECK(cudaMemcpyAsync(buffers[indices[INPUT_IDX]], data[INPUT_IDX], gSizes[INPUT_IDX] * sizeof(float), cudaMemcpyHostToDevice, stream));
    // CHECK(cudaMemcpyAsync(buffers[indices[HIDDEN_IN_IDX]], data[HIDDEN_IN_IDX], gSizes[HIDDEN_IN_IDX] * sizeof(float), cudaMemcpyHostToDevice, stream));
    // CHECK(cudaMemcpyAsync(buffers[indices[CELL_IN_IDX]], data[CELL_IN_IDX], gSizes[CELL_IN_IDX] * sizeof(float), cudaMemcpyHostToDevice, stream));

    cudaEvent_t st, ed;
    float tm = 0, tm_tmp;
    float tm_min=100000, tm_max=0;

    float* signal_host;
    float* signal_device;
    CHECK(cudaHostAlloc((void**)&signal_host, sizeof(float), cudaHostAllocPortable));
    CHECK(cudaMalloc((void**)&signal_device, sizeof(float)));

    context.enqueue(BATCH_SIZE, buffers, stream, nullptr);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaProfilerStart());
    for (int _ = 0; _ < NUM_RUNS; _++)
    {
        tm_tmp = 0;
        CHECK(cudaMemcpy(signal_device, signal_host, sizeof(float), cudaMemcpyHostToDevice));
        cudaEventCreate(&st);
        cudaEventCreate(&ed);
        cudaEventRecord(st, 0);
        // Execute asynchronously
        context.enqueue(BATCH_SIZE, buffers, stream, nullptr);

        cudaEventRecord(ed, 0);
        cudaEventSynchronize(ed);
        cudaEventElapsedTime(&tm_tmp, st, ed);
        cudaEventDestroy(st);
        cudaEventDestroy(ed);
        CHECK(cudaMemcpy(signal_host, signal_device, sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "Iteration time " << tm_tmp << " ms" << std::endl;
        tm += tm_tmp;
        tm_min=std::min(tm_min,tm_tmp);
        tm_max=std::max(tm_max,tm_tmp);
    }

    CHECK(cudaDeviceSynchronize());

    //std::cout << "Total Iterations: " << NUM_RUNS << std::endl;
    //std::cout << "TensorRT lstm avg time: " << tm / NUM_RUNS << " ms" << std::endl;
    std::cout<<"Summary: [min, max, mean] = ["<<tm_min<<", "<<tm_max<<", "<<tm/NUM_RUNS<<"] ms"<<std::endl;
    CHECK(cudaProfilerStop());

    // DMA the output from the GPU
    // CHECK(cudaMemcpyAsync(data[HIDDEN_OUT_IDX], buffers[indices[HIDDEN_OUT_IDX]], gSizes[HIDDEN_OUT_IDX] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    // CHECK(cudaMemcpyAsync(data[CELL_OUT_IDX], buffers[indices[CELL_OUT_IDX]], gSizes[CELL_OUT_IDX] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    // CHECK(cudaMemcpyAsync(data[OUTPUT_IDX], buffers[indices[OUTPUT_IDX]], gSizes[OUTPUT_IDX] * sizeof(float), cudaMemcpyDeviceToHost, stream));
}

//!
//! \brief Given an input string, this function seeds the model and then generates the expected string.
//!
//! \param context The TensorRT context used to run the model.
//! \param input The input string used to seed the model.
//! \param expected The string that is expected to be generated by the model.
//! \param weightMap Map that contains all the weights required by the model.
//!
bool doInference(IExecutionContext &context, std::string input, std::string expected, std::map<std::string, Weights> &weightMap)
{
    const ICudaEngine &engine = context.getEngine();
    assert(engine.getNbBindings() == NUM_BINDINGS);
    void *buffers[NUM_BINDINGS];
    float *data[NUM_BINDINGS];
    int indices[NUM_BINDINGS];

    std::fill(buffers, buffers + NUM_BINDINGS, nullptr);
    std::fill(data, data + NUM_BINDINGS, nullptr);
    std::fill(indices, indices + NUM_BINDINGS, -1);

    // allocate memory on host and device
    allocateMemory(engine, buffers, data, indices);

    // create stream for trt execution
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // auto embed = weightMap[EMBED_NAME];
    // float *embed_values = (float *)malloc(sizeof(float) * 65 * DATA_SIZE);
    std::string genstr;

    // Set sequence lengths to maximum
    std::fill_n(reinterpret_cast<int32_t *>(data[SEQ_LEN_IN_IDX]), BATCH_SIZE, SEQ_SIZE);
    std::cout << "seqLen: " << *reinterpret_cast<int32_t *>(data[SEQ_LEN_IN_IDX]) << std::endl;

    // Seed the RNN with the input.
    for (int i = 0; i < 1; i++)
    {
        // std::copy(static_cast<const float *>(embed_values) + char_to_id[a] * DATA_SIZE,
        //           static_cast<const float *>(embed_values) + char_to_id[a] * DATA_SIZE + DATA_SIZE,
        //           data[INPUT_IDX]);

        stepOnce(data, buffers, indices, stream, context);
        cudaStreamSynchronize(stream);

        // Copy Ct/Ht to the Ct-1/Ht-1 slots.
        // std::memcpy(data[HIDDEN_IN_IDX], data[HIDDEN_OUT_IDX], gSizes[HIDDEN_IN_IDX] * sizeof(float));
        // std::memcpy(data[CELL_IN_IDX], data[CELL_OUT_IDX], gSizes[CELL_IN_IDX] * sizeof(float));

        // genstr.push_back(a);
    }

    // Extract first predicted character
    // uint32_t predIdx = *reinterpret_cast<uint32_t *>(data[OUTPUT_IDX]);
    // genstr.push_back(id_to_char[predIdx]);

    // Generate predicted sequence of characters
    // for (size_t x = 0, y = expected.size() - 1; x < y; x++)
    // {
    //     std::copy(static_cast<const float *>(embed_values) + char_to_id[*genstr.rbegin()] * DATA_SIZE,
    //               static_cast<const float *>(embed_values) + char_to_id[*genstr.rbegin()] * DATA_SIZE + DATA_SIZE,
    //               data[INPUT_IDX]);

    //     stepOnce(data, buffers, indices, stream, context);
    //     cudaStreamSynchronize(stream);

    //     // Copy Ct/Ht to the Ct-1/Ht-1 slots.
    //     // std::memcpy(data[HIDDEN_IN_IDX], data[HIDDEN_OUT_IDX], gSizes[HIDDEN_IN_IDX] * sizeof(float));
    //     // std::memcpy(data[CELL_IN_IDX], data[CELL_OUT_IDX], gSizes[CELL_IN_IDX] * sizeof(float));

    //     // predIdx = *reinterpret_cast<uint32_t *>(data[OUTPUT_IDX]);
    //     // genstr.push_back(id_to_char[predIdx]);
    // }
    // gLogInfo << "Received: "
    //          << "passed" << std::endl;

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    for (int x = 0; x < NUM_BINDINGS; ++x)
    {
        if (!gHostInput[x])
            cudaFree(buffers[indices[x]]);
        delete[] data[x];
        data[x] = nullptr;
    }
    // return genstr == (input + expected);
    return true;
}

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_char_rnn [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use data/samples/char-rnn/ and data/char-rnn/" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
}

//!
//! \brief Runs the char-rnn model in TensorRT with a set of expected input and output strings.
//!
int main(int argc, char **argv)
{
    CHECK(cudaProfilerStop());
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gArgs.dataDirs = std::vector<std::string>{"data/samples/char-rnn/", "data/char-rnn/"};
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char **>(argv));

    // gLogger.reportTestStart(sampleTest);

    // BATCH_SIZE needs to equal one because the doInference() function
    // assumes that the batch size is one. To change this, one would need to add code to the
    // doInference() function to seed BATCH_SIZE number of inputs and process the
    // generation of BATCH_SIZE number of outputs. We leave this as an exercise for the user.
    assert(BATCH_SIZE == 1 && "This code assumes batch size is equal to 1.");

    // create a model using the API directly and serialize it to a stream
    IHostMemory *modelStream{nullptr};

    // Load weights and create model
    // std::map<std::string, Weights> weightMap = loadWeights("char-rnn.wts", weightNames);
    std::map<std::string, Weights> weightMap;
    APIToModel(weightMap, &modelStream);
    assert(modelStream != nullptr);

    // Input strings and their respective expected output strings
    // const vector<string> inS{
    //     "ROMEO",
    //     "JUL",
    //     "The K",
    //     "That tho",
    //     "KING",
    //     "beauty of",
    //     "birth of K",
    //     "Hi",
    //     "JACK",
    //     "interestingly, it was J",
    // };
    // const vector<string> outS{
    //     ":\nThe sense to",
    //     "IET:\nWhat shall I shall be",
    //     "ing Richard shall be the strange",
    //     "u shalt be the",
    //     " HENRY VI:\nWhat",
    //     " the son,",
    //     "ing Richard's son",
    //     "ng of York,\nThat thou hast so the",
    //     "INGHAM:\nWhat shall I",
    //     "uliet",
    // };

    // Select a random seed string.
    srand(unsigned(time(nullptr)));
    // int num = rand() % inS.size();

    // Initialize engine, context, and other runtime resources
    IRuntime *runtime = createInferRuntime(gLogger.getTRTLogger());
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);
    assert(engine != nullptr);
    modelStream->destroy();
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Perform inference
    bool pass = false;
    // gLogInfo << "RNN Warmup: " << inS[num] << std::endl;
    // gLogInfo << "Expect: " << outS[num] << std::endl;
    pass = doInference(*context, "", "", weightMap);

    CHECK(cudaDeviceSynchronize());

    // Clean up runtime resources
    for (auto &mem : weightMap)
        free((void *)(mem.second.values));
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // return gLogger.reportTest(sampleTest, pass);
    return 0;
}
