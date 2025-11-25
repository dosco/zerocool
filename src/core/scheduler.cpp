#include "core/scheduler.hpp"
#include <algorithm>
#include <iostream>

namespace freellm {

Scheduler::Scheduler(KVCacheManager* kv_manager) : kv_manager_(kv_manager) {}

void Scheduler::add_request(std::unique_ptr<Sequence> seq) {
    waiting_queue_.push_back(std::move(seq));
}

std::vector<Sequence*> Scheduler::step() {
    // 1. Egress: Evict finished requests
    auto it = running_queue_.begin();
    while (it != running_queue_.end()) {
        if ((*it)->is_finished()) {
            // Move to finished queue
            finished_queue_.push_back(std::move(*it));
            it = running_queue_.erase(it);
        } else {
            ++it;
        }
    }

    // 2. Ingress: Schedule new requests
    // Calculate available blocks
    size_t free_blocks = kv_manager_->free_blocks_count();
    size_t block_size = kv_manager_->config().block_size;

    // We need to reserve space for currently running requests to grow
    // Each running request needs at most 1 new block (if it crosses block boundary)
    // But wait, PagedAttention allocates on demand.
    // So we should check if running requests need a new block *now*?
    // Or just be conservative.
    // Conservative: Assume every running request might need 1 block.
    size_t needed_for_running = 0;
    for (const auto& seq : running_queue_) {
        // If (len % block_size == 0), it will need a new block for the NEXT token.
        if (seq->length() % block_size == 0) {
            needed_for_running++;
        }
    }

    if (needed_for_running > free_blocks) {
        // We are in trouble, we might OOM.
        // In a real system we might preempt.
        // For now, just don't schedule anything new.
        // And maybe we should warn.
    }
    
    size_t available_for_new = (free_blocks > needed_for_running) ? (free_blocks - needed_for_running) : 0;

    while (!waiting_queue_.empty()) {
        Sequence* candidate = waiting_queue_.front().get();
        size_t required_blocks = estimate_memory_requirement(candidate);

        if (required_blocks <= available_for_new) {
            // Move to running
            running_queue_.push_back(std::move(waiting_queue_.front()));
            waiting_queue_.pop_front();
            available_for_new -= required_blocks;
        } else {
            // Not enough memory for this request.
            // Stop scheduling to preserve FCFS (or could skip if we wanted to fill holes)
            break; 
        }
    }

    // 3. Collect batch
    std::vector<Sequence*> batch;
    for (const auto& seq : running_queue_) {
        batch.push_back(seq.get());
    }
    
    return batch;
}

std::vector<std::unique_ptr<Sequence>> Scheduler::get_finished_requests() {
    std::vector<std::unique_ptr<Sequence>> result;
    result.swap(finished_queue_);
    return result;
}

bool Scheduler::has_unfinished_requests() const {
    return !waiting_queue_.empty() || !running_queue_.empty();
}

size_t Scheduler::estimate_memory_requirement(const Sequence* seq) const {
    size_t block_size = kv_manager_->config().block_size;
    size_t len = seq->length();
    size_t blocks_per_layer = (len + block_size - 1) / block_size;
    return blocks_per_layer * seq->n_layers();
}

} // namespace freellm
