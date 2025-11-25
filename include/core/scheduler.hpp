#pragma once

#include "core/sequence.hpp"
#include "core/paged_kv_cache.hpp"
#include <deque>
#include <vector>
#include <memory>
#include <list>

namespace freellm {

/**
 * @brief Scheduler for Continuous Batching
 * 
 * Manages the lifecycle of requests (sequences) and schedules them for execution
 * based on memory availability and other constraints.
 */
class Scheduler {
public:
    Scheduler(KVCacheManager* kv_manager);

    /**
     * @brief Add a new request to the scheduler
     * 
     * @param seq The sequence to add
     */
    void add_request(std::unique_ptr<Sequence> seq);

    /**
     * @brief Perform a single scheduling step
     * 
     * This function:
     * 1. Evicts finished requests from the running queue.
     * 2. Schedules new requests from the waiting queue if memory allows.
     * 3. Returns the list of sequences to be executed in the current step.
     * 
     * @return List of pointers to sequences that should run in this step
     */
    std::vector<Sequence*> step();

    /**
     * @brief Get requests that have finished execution
     * 
     * @return List of finished sequences (ownership transferred to caller)
     */
    std::vector<std::unique_ptr<Sequence>> get_finished_requests();

    /**
     * @brief Check if there are any unfinished requests (waiting or running)
     */
    bool has_unfinished_requests() const;

private:
    // Estimate memory blocks needed for a sequence in the next step
    size_t estimate_memory_requirement(const Sequence* seq) const;

    KVCacheManager* kv_manager_;
    
    // Request queues
    std::deque<std::unique_ptr<Sequence>> waiting_queue_;
    std::list<std::unique_ptr<Sequence>> running_queue_;
    std::vector<std::unique_ptr<Sequence>> finished_queue_;
};

} // namespace freellm
