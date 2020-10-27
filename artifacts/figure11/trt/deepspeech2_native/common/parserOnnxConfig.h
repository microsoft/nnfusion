/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef PARSER_ONNX_CONFIG_H
#define PARSER_ONNX_CONFIG_H

#include <cstring>
#include <iostream>
#include <string>

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

#define ONNX_DEBUG 1

/**
 * \class ParserOnnxConfig
 * \brief Configuration Manager Class Concrete Implementation
 *
 * \note:
 *
 */

using namespace std;

class ParserOnnxConfig : public nvonnxparser::IOnnxConfig
{

protected:
    string mModelFilename{};
    string mTextFilename{};
    string mFullTextFilename{};
    nvinfer1::DataType mModelDtype;
    nvonnxparser::IOnnxConfig::Verbosity mVerbosity;
    bool mPrintLayercInfo;

public:
    ParserOnnxConfig()
        : mModelDtype(nvinfer1::DataType::kFLOAT)
        , mVerbosity(static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))
        , mPrintLayercInfo(false)
    {
#ifdef ONNX_DEBUG
        if (isDebug())
        {
            std::cout << " ParserOnnxConfig::ctor(): "
                      << this << "\t"
                      << std::endl;
        }
#endif
    }

protected:
    ~ParserOnnxConfig()
    {
#ifdef ONNX_DEBUG
        if (isDebug())
        {
            std::cout << "ParserOnnxConfig::dtor(): " << this << std::endl;
        }
#endif
    }

public:
    virtual void setModelDtype(const nvinfer1::DataType modelDtype) { mModelDtype = modelDtype; }

    virtual nvinfer1::DataType getModelDtype() const
    {
        return mModelDtype;
    }

    virtual const char* getModelFileName() const { return mModelFilename.c_str(); }
    virtual void setModelFileName(const char* onnxFilename)
    {
        mModelFilename = string(onnxFilename);
    }
    virtual nvonnxparser::IOnnxConfig::Verbosity getVerbosityLevel() const { return mVerbosity; }
    virtual void addVerbosity() { ++mVerbosity; }
    virtual void reduceVerbosity() { --mVerbosity; }
    virtual void setVerbosityLevel(nvonnxparser::IOnnxConfig::Verbosity verbosity) { mVerbosity = verbosity; }

    virtual const char* getTextFileName() const { return mTextFilename.c_str(); }
    virtual void setTextFileName(const char* textFilename)
    {
        mTextFilename = string(textFilename);
    }
    virtual const char* getFullTextFileName() const { return mFullTextFilename.c_str(); }
    virtual void setFullTextFileName(const char* fullTextFilename)
    {
        mFullTextFilename = string(fullTextFilename);
    }
    virtual bool getPrintLayerInfo() const { return mPrintLayercInfo; }
    virtual void setPrintLayerInfo(bool src) { mPrintLayercInfo = src; } //!< get the boolean variable corresponding to the Layer Info, see getPrintLayerInfo()

    virtual bool isDebug() const
    {
#if ONNX_DEBUG
        return (std::getenv("ONNX_DEBUG") ? true : false);
#else
        return false;
#endif
    }

    virtual void destroy() { delete this; }

}; // class ParserOnnxConfig

#endif
