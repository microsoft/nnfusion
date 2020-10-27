#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include "NvInfer.h"
#include "common.h"
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <vector>

class BatchStream
{
public:
    BatchStream(int batchSize, int maxBatches, std::string prefix, std::vector<std::string> directories)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mPrefix(prefix)
        , mDataDir(directories)
    {
        FILE* file = fopen(locateFile(mPrefix + std::string("0.batch"), mDataDir).c_str(), "rb");
        int d[4];
        size_t readSize = fread(d, sizeof(int), 4, file);
        assert(readSize == 4);
        mDims.nbDims = 4;  //The number of dimensions.
        mDims.d[0] = d[0]; //Batch Size
        mDims.d[1] = d[1]; //Channels
        mDims.d[2] = d[2]; //Height
        mDims.d[3] = d[3]; //Width

        fclose(file);
        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        reset(0);
    }

    // Resets data members
    void reset(int firstBatch)
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.d[0];
        skip(firstBatch);
    }

    // Advance to next batch and return true, or return false if there is no batch left.
    bool next()
    {
        if (mBatchCount == mMaxBatches)
            return false;

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.d[0]);
            if (mFileBatchPos == mDims.d[0] && !update())
                return false;

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.d[0] - mFileBatchPos);
            std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
        }
        mBatchCount++;
        return true;
    }

    // Skips the batches
    void skip(int skipCount)
    {
        if (mBatchSize >= mDims.d[0] && mBatchSize % mDims.d[0] == 0 && mFileBatchPos == mDims.d[0])
        {
            mFileCount += skipCount * mBatchSize / mDims.d[0];
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
            next();
        mBatchCount = x;
    }

    float* getBatch() { return &mBatch[0]; }
    int getBatchesRead() const { return mBatchCount; }
    int getBatchSize() const { return mBatchSize; }
    int getImageSize() const { return mImageSize; }
    nvinfer1::Dims getDims() const { return mDims; }

private:
    float* getFileBatch() { return &mFileBatch[0]; }

    bool update()
    {
        std::string inputFileName = locateFile(mPrefix + std::to_string(mFileCount++) + std::string(".batch"), mDataDir);
        FILE* file = fopen(inputFileName.c_str(), "rb");
        if (!file)
            return false;

        int d[4];
        size_t readSize = fread(d, sizeof(int), 4, file);
        assert(readSize == 4);
        assert(mDims.d[0] == d[0] && mDims.d[1] == d[1] && mDims.d[2] == d[2] && mDims.d[3] == d[3]);
        size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.d[0] * mImageSize, file);
        assert(readInputCount == size_t(mDims.d[0] * mImageSize));

        fclose(file);
        mFileBatchPos = 0;
        return true;
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    int mFileCount{0};
    int mFileBatchPos{0};
    int mImageSize{0};
    nvinfer1::Dims mDims;
    std::vector<float> mBatch;
    std::vector<float> mFileBatch;
    std::string mPrefix;
    std::vector<std::string> mDataDir;
};
#endif
