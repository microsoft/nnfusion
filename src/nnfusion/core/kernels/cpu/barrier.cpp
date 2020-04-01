// Microsoft (c) 2019, NNFUSION Team
#include "barrier.hpp"

namespace nnfusion
{
    namespace kernels
    {
        LanguageUnit_p barrier_header = LanguageUnit_p(new LanguageUnit("barrier.h",
                                                                        R"(

#pragma once

#include <assert.h>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>

namespace nnfusion
{
    namespace cpu
    {
        // Barrier is an object that allows one or more threads to wait until
        // Notify has been called a specified number of times.
        class Barrier 
        {
        public:
            Barrier(unsigned int count) 
            : count_(count)
            , state_(count << 1)
            , notified_(false) 
            {
                assert(((count << 1) >> 1) == count);
            }
            ~Barrier() { assert((state_ >> 1) == 0); }

            void Notify() 
            {
                unsigned int v = state_.fetch_sub(2, std::memory_order_acq_rel) - 2;
                if (v != 1) 
                {
                    // Clear the lowest bit (waiter flag) and check that the original state
                    // value was not zero. If it was zero, it means that notify was called
                    // more times than the original count.
                    assert(((v + 2) & ~1) != 0);
                    return;  // either count has not dropped to 0, or waiter is not waiting
                }
                std::unique_lock<std::mutex> l(mu_);
                assert(!notified_);
                notified_ = true;
                cv_.notify_all();
            }

            void Wait() 
            {
                unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
                if ((v >> 1) == 0) return;
                std::unique_lock<std::mutex> l(mu_);
                while (!notified_) 
                {
                    cv_.wait(l);
                }
            }

            void Reset()
            {
                state_ = (count_ << 1);
                notified_ = false;
            }

        private:
            std::mutex mu_;
            std::condition_variable cv_;
            std::atomic<unsigned int> state_;  // low bit is waiter flag
            unsigned int count_;  // for Reset()
            bool notified_;
        };

        // Notification is an object that allows a user to to wait for another
        // thread to signal a notification that an event has occurred.
        //
        // Multiple threads can wait on the same Notification object,
        // but only one caller must call Notify() on the object.
        struct Notification : Barrier 
        {
            Notification() : Barrier(1){};
        }; 
    }   
}
)"));
    }
}