// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <random>

namespace nnfusion
{
    namespace test
    {
        /// \brief A predictable pseudo-random number generator
        /// The seed is initialized so that we get repeatable pseudo-random numbers for tests
        template <typename T>
        class Uniform
        {
        public:
            Uniform(T min, T max, T seed = 0)
                : m_engine(seed)
                , m_distribution(min, max)
                , m_r(std::bind(m_distribution, m_engine))
            {
            }

            /// \brief Randomly initialize a tensor
            /// \param ptv The tensor to initialize
            const std::shared_ptr<runtime::Tensor>
                initialize(const std::shared_ptr<runtime::Tensor>& ptv)
            {
                std::vector<T> vec = read_vector<T>(ptv);
                initialize(vec);
                write_vector(ptv, vec);
                return ptv;
            }
            /// \brief Randomly initialize a vector
            /// \param vec The tensor to initialize
            void initialize(std::vector<T>& vec)
            {
                for (T& elt : vec)
                {
                    elt = m_r();
                }
            }

        protected:
            std::default_random_engine m_engine;
            std::uniform_real_distribution<T> m_distribution;
            std::function<T()> m_r;
        };
    }
}
