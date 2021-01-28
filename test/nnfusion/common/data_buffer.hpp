// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <climits>
#include <cstdlib>
#include <vector>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "nnfusion/common/type/data_buffer.hpp"
#include "nnfusion/common/type/element_type.hpp"

namespace nnfusion
{
    namespace test
    {
        template <typename T>
        void random_data(T* arr, size_t len)
        {
            for (size_t i = 0; i < len; ++i)
            {
                arr[i] = rand();
            }
        }

        template <>
        void random_data<element::half>(element::half* arr, size_t len)
        {
            for (size_t i = 0; i < len; ++i)
            {
                arr[i] = 1. * rand() / RAND_MAX;
            }
        }

        template <typename T, typename U, typename V>
        bool arrcmp(U* x, V* y, size_t len)
        {
            T* lhs = reinterpret_cast<T*>(x);
            T* rhs = reinterpret_cast<T*>(y);
            for (size_t i = 0; i < len; ++i)
            {
                if (lhs[i] != rhs[i])
                {
                    NNFUSION_LOG(INFO) << "lhs[" << i << "] = " << lhs[i] << " != rhs[" << i
                                       << "] = " << rhs[i];
                    return false;
                }
            }
            return true;
        }

        template <typename U, typename V>
        bool bitcmp(U* x, V* y, size_t len)
        {
            uint8_t* lhs = reinterpret_cast<uint8_t*>(x);
            uint8_t* rhs = reinterpret_cast<uint8_t*>(y);
            for (size_t i = 0; i < len; ++i)
            {
                if (lhs[i] != rhs[i])
                {
                    NNFUSION_LOG(INFO) << "lhs[" << i << "] = " << lhs[i] << " != rhs[" << i
                                       << "] = " << rhs[i];
                    return false;
                }
            }
            return true;
        }

        template <typename T>
        bool bufarrcmp(const DataBuffer& buf, T* arr)
        {
            for (size_t i = 0; i < buf.size(); ++i)
            {
                T x;
                T y = arr[i];
                buf.getElement(i, &x);
                if (x != y)
                {
                    NNFUSION_LOG(INFO) << "lhs[" << i << "] = " << x << " != rhs[" << i
                                       << "] = " << y;
                    return false;
                }
            }
            return true;
        }

        template <typename T>
        void basicTestDataBuffer()
        {
            element::Type tp = element::from<T>();
            DataBuffer buf(tp);
            T ele[10];

            NNFUSION_LOG(INFO) << "Testting DataBuffer for type " << tp;

            // test constructor
            ASSERT_EQ(buf.size(), 0U);
            ASSERT_EQ(buf.size_in_bytes(), 0U);
            ASSERT_EQ(buf.data(), nullptr);
            ASSERT_EQ(buf.get_type(), tp);

            // test setElement & getElement, data()
            buf.resize(10);
            ASSERT_EQ(buf.size(), 10U);
            ASSERT_EQ(buf.size_in_bytes(), 10U * sizeof(T));
            ASSERT_NE(buf.data(), nullptr);
            for (size_t i = 0; i < buf.size(); ++i)
            {
                buf.setElement(i, ele + i);
            }
            T* dat = reinterpret_cast<T*>(buf.data());
            for (size_t i = 0; i < buf.size(); ++i)
            {
                T x;
                // ASSERT_EQ(dat[i], ele[i]);
                buf.getElement(i, &x);
                ASSERT_EQ(ele[i], x);
            }

            // test move_to
            void *p = nullptr, *pt = buf.data();
            buf.move_to(reinterpret_cast<void**>(&p));
            ASSERT_NE(p, nullptr);
            ASSERT_EQ(buf.data(), nullptr);
            ASSERT_EQ(buf.size(), 0U);
            ASSERT_EQ(p, pt);

            // test load
            buf.load(p, 10);
            ASSERT_TRUE(bitcmp(buf.data(), p, 10 * tp.size()));

            // test dump
            memset(p, 0, 10 * tp.size());
            buf.dump(p);
            ASSERT_TRUE(bitcmp(buf.data(), p, 10 * tp.size()));

            // test copy constructor
            DataBuffer buf2(buf);
            ASSERT_EQ(buf.size(), buf2.size());
            ASSERT_NE(buf.data(), buf2.data());
            ASSERT_EQ(buf.get_type(), buf2.get_type());
            ASSERT_TRUE(bitcmp(buf.data(), buf2.data(), 10 * tp.size()));

            // test move constructor
            DataBuffer buf3(std::move(buf2));
            ASSERT_EQ(buf2.size(), 0U);
            ASSERT_EQ(buf2.data(), nullptr);
            ASSERT_EQ(buf3.size(), 10U);
            ASSERT_NE(buf3.data(), nullptr);
            ASSERT_TRUE(bitcmp(buf3.data(), buf.data(), 10 * tp.size()));

            // test loadVector
            buf.loadVector(std::vector<int>{1, 2, 3, 4, 5});
            buf2.loadVector(std::vector<short>{1, 2, 3, 4, 5});
            ASSERT_EQ(buf.size(), 5U);
            ASSERT_EQ(buf2.size(), 5U);
            ASSERT_TRUE(arrcmp<T>(buf.data(), buf2.data(), 5));
        }

        template <typename T>
        void testStringLoading()
        {
            element::Type tp = element::from<T>();
            DataBuffer buf(tp), buf2(tp), buf3(tp);
            T ele[10];
            // test loadFromStrings
            buf.loadVector(std::vector<int>{1, 2, 3, 4, 5});
            buf3.loadFromStrings(std::vector<std::string>{"1", "2", "3", "4", "5"});
            ASSERT_EQ(buf3.size(), 5U);
            ASSERT_TRUE(arrcmp<T>(buf3.data(), buf.data(), 5));
            buf.resize(10);
            for (size_t i = 0; i < 10; ++i)
            {
                T x = T(3);
                buf.setElement(i, &x);
            }
            buf2.loadFromStrings({"3"}, 10);
            ASSERT_EQ(buf2.size(), 10U);
            for (size_t i = 0; i < 10; ++i)
            {
                T x, y;
                buf.getElement(i, &x);
                buf2.getElement(i, &y);
                ASSERT_EQ(x, y);
            }
        }

        template <>
        void testStringLoading<int8_t>()
        {
        }

        template <>
        void testStringLoading<uint8_t>()
        {
        }

        template <>
        void testStringLoading<char>()
        {
        }

        template <typename T>
        void fullTestDataBuffer()
        {
            basicTestDataBuffer<T>();
            testStringLoading<T>();
        }
    }
}
