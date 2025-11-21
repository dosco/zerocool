#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>

namespace freellm {

/**
 * @brief Thread-safe bounded queue for producer-consumer pattern
 *
 * This queue implements backpressure: when full, producers will block
 * until consumers make room. This prevents unbounded memory growth.
 *
 * Key features:
 * - Thread-safe push/pop operations
 * - Blocking push when queue is full (backpressure)
 * - Blocking pop when queue is empty
 * - Graceful shutdown mechanism
 * - Move semantics for efficient data transfer
 *
 * Typical usage:
 *   BoundedQueue<WorkItem> queue(capacity);
 *
 *   // Producer thread:
 *   queue.push(std::move(item));
 *   queue.shutdown(); // Signal completion
 *
 *   // Consumer threads:
 *   while (auto item = queue.pop()) {
 *       process(*item);
 *   }
 */
template<typename T>
class BoundedQueue {
public:
    /**
     * @brief Construct bounded queue with maximum capacity
     *
     * @param capacity Maximum number of items in queue
     */
    explicit BoundedQueue(size_t capacity)
        : capacity_(capacity), shutdown_(false) {}

    /**
     * @brief Push item into queue (blocks if full)
     *
     * Blocks until there's room in the queue, or until shutdown is called.
     *
     * @param item Item to push (will be moved)
     * @return true if item was pushed, false if queue is shutdown
     */
    bool push(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait until queue has room or shutdown
        not_full_.wait(lock, [this] {
            return queue_.size() < capacity_ || shutdown_;
        });

        // Don't accept new items after shutdown
        if (shutdown_) {
            return false;
        }

        queue_.push(std::move(item));

        // Notify waiting consumers
        not_empty_.notify_one();

        return true;
    }

    /**
     * @brief Pop item from queue (blocks if empty)
     *
     * Blocks until an item is available, or until queue is shutdown
     * and empty.
     *
     * @return Item if available, std::nullopt if queue is shutdown and empty
     */
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait until queue has items or is shutdown
        not_empty_.wait(lock, [this] {
            return !queue_.empty() || shutdown_;
        });

        // If shutdown and empty, we're done
        if (queue_.empty() && shutdown_) {
            return std::nullopt;
        }

        // Get item from queue
        T item = std::move(queue_.front());
        queue_.pop();

        // Notify waiting producers
        not_full_.notify_one();

        return item;
    }

    /**
     * @brief Try to pop item with timeout
     *
     * @param timeout_ms Timeout in milliseconds
     * @return Item if available, std::nullopt if timeout or shutdown
     */
    std::optional<T> try_pop(int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait with timeout
        if (!not_empty_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                 [this] { return !queue_.empty() || shutdown_; })) {
            return std::nullopt;  // Timeout
        }

        // If shutdown and empty, we're done
        if (queue_.empty() && shutdown_) {
            return std::nullopt;
        }

        // Get item from queue
        T item = std::move(queue_.front());
        queue_.pop();

        // Notify waiting producers
        not_full_.notify_one();

        return item;
    }

    /**
     * @brief Signal that no more items will be pushed
     *
     * After calling shutdown(), push() will return false.
     * Consumer threads will finish processing existing items
     * and then pop() will return std::nullopt.
     */
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }

        // Wake up all waiting threads
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    /**
     * @brief Get current queue size
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /**
     * @brief Check if queue is empty
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /**
     * @brief Check if shutdown was called
     */
    bool is_shutdown() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return shutdown_;
    }

private:
    size_t capacity_;                    // Maximum queue size
    std::queue<T> queue_;                // Underlying queue
    mutable std::mutex mutex_;           // Protects queue and shutdown flag
    std::condition_variable not_full_;   // Signaled when queue has room
    std::condition_variable not_empty_;  // Signaled when queue has items
    bool shutdown_;                      // True after shutdown() called
};

} // namespace freellm
