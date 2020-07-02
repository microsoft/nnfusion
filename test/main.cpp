// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <chrono>
#include <iostream>

#include "gtest/gtest.h"

using namespace std;

#ifdef NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

int main(int argc, char** argv)
{
#ifdef NGRAPH_DISTRIBUTED
    ngraph::Distributed dist;
#endif
    const char* exclude = "--gtest_filter=-benchmark.*";
    vector<char*> argv_vector;
    argv_vector.push_back(argv[0]);
    argv_vector.push_back(const_cast<char*>(exclude));
    for (int i = 1; i < argc; i++)
    {
        argv_vector.push_back(argv[i]);
    }
    argc++;

    ::testing::InitGoogleTest(&argc, argv_vector.data());
    int rc = RUN_ALL_TESTS();

    return rc;
}
