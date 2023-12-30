#pragma once

#include <iostream>
#include <chrono>

#define KERNEL_TIMER

class Timer {
public:
    using time_type = std::chrono::microseconds;

    static Timer &get_instance() {
         static Timer timer;
         return timer;
    }

    static void total_start() {
        auto& timer = get_instance();
        timer.total_times_[0] = std::chrono::high_resolution_clock::now();
    }

    static void total_stop() {
        auto& timer = get_instance();
        timer.total_times_[1] = std::chrono::high_resolution_clock::now();
    }

    static void kernel_start() {
#ifdef KERNEL_TIMER
        auto& timer = get_instance();
        timer.kernel_times_[0] = std::chrono::high_resolution_clock::now();
#endif
    }

    static void kernel_stop() {
#ifdef KERNEL_TIMER
        auto& timer = get_instance();
        timer.kernel_times_[1] = std::chrono::high_resolution_clock::now();
#endif
    }

    static int64_t total_cost() {
        auto& times = get_instance().total_times_;
        return std::chrono::duration_cast<time_type>(times[1] - times[0])
                         .count();
    }

    static int64_t kernel_cost() {
#ifdef KERNEL_TIMER
        auto& times = get_instance().kernel_times_;
        return std::chrono::duration_cast<time_type>(times[1] - times[0])
                         .count();
#elif
        return 0;
#endif
    }

    Timer(const Timer &single) = delete;
    const Timer &operator=(const Timer &single) = delete;

private:
    Timer() = default;

    std::chrono::high_resolution_clock::time_point total_times_[2]{};
#ifdef KERNEL_TIMER
    std::chrono::high_resolution_clock::time_point kernel_times_[2]{};
#endif
};
