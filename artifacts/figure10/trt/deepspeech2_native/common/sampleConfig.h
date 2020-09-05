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

#ifndef SampleConfig_H
#define SampleConfig_H

#include <cstring>
#include <iostream>
#include <string>

#include "NvInfer.h"
#include "NvOnnxConfig.h"
class SampleConfig : public nvonnxparser::IOnnxConfig
{
public:
    enum class InputDataFormat : int
    {
        kASCII = 0,
        kPPM = 1
    };

private:
    std::string mModelFilename;
    std::string mEngineFilename;
    std::string mTextFilename;
    std::string mFullTextFilename;
    std::string mImageFilename;
    std::string mReferenceFilename;
    std::string mOutputFilename;
    std::string mCalibrationFilename;
    int64_t mMaxBatchSize;
    int64_t mMaxWorkspaceSize;
    int64_t mCalibBatchSize;
    int64_t mMaxNCalibBatch;
    int64_t mFirstCalibBatch;
    int64_t mUseDLACore;
    nvinfer1::DataType mModelDtype;
    Verbosity mVerbosity;
    bool mPrintLayercInfo;
    bool mDebugBuilder;
    InputDataFormat mInputDataFormat;
    uint64_t mTopK;

public:
    SampleConfig()
        : mMaxBatchSize(32)
        , mMaxWorkspaceSize(1 * 1024 * 1024 * 1024)
        , mCalibBatchSize(0)
        , mMaxNCalibBatch(0)
        , mFirstCalibBatch(0)
        , mUseDLACore(-1)
        , mModelDtype(nvinfer1::DataType::kFLOAT)
        , mVerbosity(static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))
        , mPrintLayercInfo(false)
        , mDebugBuilder(false)
        , mInputDataFormat(InputDataFormat::kASCII)
        , mTopK(0)
    {
#ifdef ONNX_DEBUG
        if (isDebug())
        {
            std::cout << " SampleConfig::ctor(): "
                      << this << "\t"
                      << std::endl;
        }
#endif
    }

protected:
    ~SampleConfig()
    {
#ifdef ONNX_DEBUG
        if (isDebug())
        {
            std::cout << "SampleConfig::dtor(): " << this << std::endl;
        }
#endif
    }

public:
    void setModelDtype(const nvinfer1::DataType mdt) { mModelDtype = mdt; }

    nvinfer1::DataType getModelDtype() const
    {
        return mModelDtype;
    }

    const char* getModelFileName() const { return mModelFilename.c_str(); }

    void setModelFileName(const char* onnxFilename)
    {
        mModelFilename = string(onnxFilename);
    }
    Verbosity getVerbosityLevel() const { return mVerbosity; }
    void addVerbosity() { ++mVerbosity; }
    void reduceVerbosity() { --mVerbosity; }
    virtual void setVerbosityLevel(Verbosity v) { mVerbosity = v; }
    const char* getEngineFileName() const { return mEngineFilename.c_str(); }
    void setEngineFileName(const char* engineFilename)
    {
        mEngineFilename = string(engineFilename);
    }
    const char* getTextFileName() const { return mTextFilename.c_str(); }
    void setTextFileName(const char* textFilename)
    {
        mTextFilename = string(textFilename);
    }
    const char* getFullTextFileName() const { return mFullTextFilename.c_str(); }
    void setFullTextFileName(const char* fullTextFilename)
    {
        mFullTextFilename = string(fullTextFilename);
    }
    bool getPrintLayerInfo() const { return mPrintLayercInfo; }
    void setPrintLayerInfo(bool b) { mPrintLayercInfo = b; } //!< get the boolean variable corresponding to the Layer Info, see getPrintLayerInfo()

    void setMaxBatchSize(int64_t maxBatchSize) { mMaxBatchSize = maxBatchSize; } //!<  set the Max Batch Size
    int64_t getMaxBatchSize() const { return mMaxBatchSize; }                    //!<  get the Max Batch Size

    void setMaxWorkSpaceSize(int64_t maxWorkSpaceSize) { mMaxWorkspaceSize = maxWorkSpaceSize; } //!<  set the Max Work Space size
    int64_t getMaxWorkSpaceSize() const { return mMaxWorkspaceSize; }                            //!<  get the Max Work Space size

    void setCalibBatchSize(int64_t CalibBatchSize) { mCalibBatchSize = CalibBatchSize; } //!<  set the calibration batch size
    int64_t getCalibBatchSize() const { return mCalibBatchSize; }                        //!<  get calibration batch size

    void setMaxNCalibBatch(int64_t MaxNCalibBatch) { mMaxNCalibBatch = MaxNCalibBatch; } //!<  set Max Number of Calibration Batches
    int64_t getMaxNCalibBatch() const { return mMaxNCalibBatch; }                        //!<  get the Max Number of Calibration Batches

    void setFirstCalibBatch(int64_t FirstCalibBatch) { mFirstCalibBatch = FirstCalibBatch; } //!<  set the first calibration batch
    int64_t getFirstCalibBatch() const { return mFirstCalibBatch; }                          //!<  get the first calibration batch

    void setUseDLACore(int64_t UseDLACore) { mUseDLACore = UseDLACore; } //!<  set the DLA core to use
    int64_t getUseDLACore() const { return mUseDLACore; }                //!<  get the DLA core to use

    void setDebugBuilder() { mDebugBuilder = true; }       //!<  enable the Debug info, while building the engine.
    bool getDebugBuilder() const { return mDebugBuilder; } //!<  get the boolean variable, corresponding to the debug builder

    const char* getImageFileName() const { return mImageFilename.c_str(); } //!<  set Image file name (PPM or ASCII)
    void setImageFileName(const char* imageFilename)                        //!< get the Image file name
    {
        mImageFilename = string(imageFilename);
    }
    const char* getReferenceFileName() const { return mReferenceFilename.c_str(); }
    void setReferenceFileName(const char* referenceFilename) //!<  set reference file name
    {
        mReferenceFilename = string(referenceFilename);
    }

    void setInputDataFormat(InputDataFormat idt) { mInputDataFormat = idt; } //!<  specifies expected data format of the image file (PPM or ASCII)
    InputDataFormat getInputDataFormat() const { return mInputDataFormat; }  //!<  returns the expected data format of the image file.

    const char* getOutputFileName() const { return mOutputFilename.c_str(); } //!<  specifies the file to save the results
    void setOutputFileName(const char* outputFilename)                        //!<  get the output file name
    {
        mOutputFilename = string(outputFilename);
    }

    const char* getCalibrationFileName() const { return mCalibrationFilename.c_str(); } //!<  specifies the file containing the list of image files for int8 calibration
    void setCalibrationFileName(const char* calibrationFilename)                        //!<  get the int 8 calibration list file name
    {
        mCalibrationFilename = string(calibrationFilename);
    }

    uint64_t getTopK() const { return mTopK; }
    void setTopK(uint64_t topK) { mTopK = topK; } //!<  If this options is specified, return the K top probabilities.

    bool isDebug() const
    {
#if ONNX_DEBUG
        return (std::getenv("ONNX_DEBUG") ? true : false);
#else
        return false;
#endif
    }

    void destroy() { delete this; }

}; // class SampleConfig

#endif
