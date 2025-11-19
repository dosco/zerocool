#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <type_traits>

namespace freellm {

/**
 * @brief Simple ThreadPool for parallelizing inference tasks
 */
class ThreadPool {
public:
    // Singleton accessor
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) : stop_(false) {
        for(size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this]{ return stop_ || !tasks_.empty(); });
                        if(stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for(std::thread &worker : workers_) {
            if(worker.joinable()) worker.join();
        }
    }

    // Enqueue a generic task and return a future
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result_t<F, Args...>>
    {
        using return_type = typename std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if(stop_) throw std::runtime_error("enqueue on stopped ThreadPool");
            
            // Wrap packed_task in a void() lambda
            tasks_.emplace([task](){ (*task)(); });
        }
        condition_.notify_one();
        return res;
    }

    // Parallel for loop helper: executes func(i) for i in [start, end)
    template<typename Func>
    void parallel_for(size_t start, size_t end, Func func) {
        if (start >= end) return;

        std::vector<std::future<void>> futures;
        futures.reserve(end - start);

        for (size_t i = start; i < end; ++i) {
            futures.push_back(enqueue([func, i] {
                func(i);
            }));
        }

        // Wait for all tasks to complete
        for (auto& f : futures) {
            f.wait();
        }
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

} // namespace freellm

