#ifndef TENSORRT_ARGS_PARSER_H
#define TENSORRT_ARGS_PARSER_H

#include <vector>
#include <string>
#ifdef _MSC_VER
#include "..\common\windows\getopt.h"
#else
#include <getopt.h>
#endif
#include <iostream>

namespace samplesCommon
{

//!
//! \brief The SampleParams structure groups the basic parameters required by
//!        all sample networks.
//!
struct SampleParams
{
    int batchSize; //!< Number of inputs in a batch
    int dlaCore{-1};  //!< Specify the DLA core to run network on.
    bool int8{false}; //!< Allow runnning the network in Int8 mode.
    bool fp16{false}; //!< Allow running the network in FP16 mode.
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
};

//!
//! \brief The CaffeSampleParams structure groups the additional parameters required by
//!         networks that use caffe
//!
struct CaffeSampleParams : public SampleParams
{
    std::string prototxtFileName; //!< Filename of prototxt design file of a network
    std::string weightsFileName;  //!< Filename of trained weights file of a network
};

//!
//! /brief Struct to maintain command-line arguments.
//!
struct Args
{
    bool runInInt8{false};
    bool runInFp16{false};
    bool help{false};
    int useDLACore{-1};
    std::vector<std::string> dataDirs;
};

//!
//! \brief Populates the Args struct with the provided command-line parameters.
//!
//! \throw invalid_argument if any of the arguments are not valid
//!
//! \return boolean If return value is true, execution can continue, otherwise program should exit
//!
inline bool parseArgs(Args& args, int argc, char* argv[])
{
    while (1)
    {
        int arg;
        static struct option long_options[] = {
            {"help", no_argument, 0, 'h'},
            {"datadir", required_argument, 0, 'd'},
            {"int8", no_argument, 0, 'i'},
            {"fp16", no_argument, 0, 'f'},
            {"useDLACore", required_argument, 0, 'u'},
            {nullptr, 0, nullptr, 0}};
        int option_index = 0;
        arg = getopt_long(argc, argv, "hd:iu", long_options, &option_index);
        if (arg == -1)
            break;

        switch (arg)
        {
        case 'h':
            args.help = true;
            return false;
        case 'd':
            if (optarg)
                args.dataDirs.push_back(optarg);
            else
            {
                std::cerr << "ERROR: --datadir requires option argument" << std::endl;
                return false;
            }
            break;
        case 'i':
            args.runInInt8 = true;
            break;
        case 'f':
            args.runInFp16 = true;
            break;
        case 'u':
            if (optarg)
                args.useDLACore = std::stoi(optarg);
            break;
        default:
            return false;
        }
    }
    return true;
}

} // namespace samplesCommon

#endif // TENSORRT_ARGS_PARSER_H
