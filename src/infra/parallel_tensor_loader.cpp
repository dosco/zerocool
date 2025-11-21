#include "infra/parallel_tensor_loader.hpp"
#include "infra/safetensors.hh"
#include "infra/safetensors_loader.hpp"  // For bf16_to_f32, fp16_to_f32
#include <algorithm>
#include <stdexcept>

namespace freellm {

// Helper for safe unaligned memory access
template <typename T>
inline T read_unaligned(const uint8_t* ptr) {
    T val;
    std::memcpy(&val, ptr, sizeof(T));
    return val;
}

WeightMap ParallelTensorLoader::load(
    const std::string& filepath,
    std::function<std::string(const std::string&)> name_mapper,
    std::function<std::optional<quant::QuantType>(const std::string&)> quant_resolver
) {
    std::println("Loading safetensors with parallel pipeline: {}", filepath);
    std::println("  Worker threads: {}", num_threads_);
    std::println("  Queue capacity: {}", queue_capacity_);

    // Create work queue
    BoundedQueue<TensorWorkItem> work_queue(queue_capacity_);

    // Per-thread results storage to avoid mutex contention
    std::vector<std::vector<TensorResult>> all_results(num_threads_);

    // ================================================================
    // Start Consumer Threads
    // ================================================================
    std::println("  Starting {} consumer threads...", num_threads_);
    std::vector<std::jthread> workers;
    workers.reserve(num_threads_);

    for (size_t i = 0; i < num_threads_; ++i) {
        workers.emplace_back([this, &work_queue, &thread_results = all_results[i]]() {
            consumer_thread(work_queue, thread_results);
        });
    }

    // ================================================================
    // Run Producer (in this thread)
    // ================================================================
    std::println("  Producer: memory-mapping file and enqueueing work items...");
    try {
        producer_thread(filepath, work_queue, name_mapper, quant_resolver);
    } catch (const std::exception& e) {
        // Error in producer - shut down queue and wait for workers
        std::println(stderr, "  Producer error: {}", e.what());
        work_queue.shutdown();
        // No need to manually join workers; jthread destructor handles it
        throw;
    }

    // Producer finished - signal workers to finish
    work_queue.shutdown();
    std::println("  Producer finished, waiting for workers...");

    // ================================================================
    // Wait for All Workers
    // ================================================================
    // Although jthread joins on destruction, we need to wait explicitly here
    // to ensure results are ready before assembling them.
    for (auto& worker : workers) {
        worker.join();
    }

    std::println("  All workers finished, assembling WeightMap...");

    // ================================================================
    // Assemble Final WeightMap
    // ================================================================
    WeightMap weights;
    // Iterate through thread-local results and merge
    for (auto& thread_results : all_results) {
        for (auto& result : thread_results) {
            weights[result.name] = std::move(result.tensor);
        }
    }

    // Clear mmap handle now that all workers are done
    current_mmap_.reset();

    std::println("  ✓ Loaded {} tensors using parallel pipeline", weights.size());
    return weights;
}

std::pair<WeightMap, std::unordered_map<std::string, QuantizedTensor>>
ParallelTensorLoader::load_with_quantization(
    const std::string& filepath,
    std::function<std::string(const std::string&)> name_mapper,
    std::function<std::optional<quant::QuantType>(const std::string&)> quant_resolver
) {
    std::println("Loading safetensors with parallel pipeline + quantization: {}", filepath);
    std::println("  Worker threads: {}", num_threads_);
    std::println("  Queue capacity: {}", queue_capacity_);

    // Create work queue
    BoundedQueue<TensorWorkItem> work_queue(queue_capacity_);

    // Per-thread results storage
    std::vector<std::vector<TensorResult>> all_results(num_threads_);

    // ================================================================
    // Start Consumer Threads
    // ================================================================
    std::println("  Starting {} consumer threads...", num_threads_);
    std::vector<std::jthread> workers;
    workers.reserve(num_threads_);

    for (size_t i = 0; i < num_threads_; ++i) {
        workers.emplace_back([this, &work_queue, &thread_results = all_results[i]]() {
            consumer_thread(work_queue, thread_results);
        });
    }

    // ================================================================
    // Run Producer (in this thread)
    // ================================================================
    std::println("  Producer: memory-mapping file and enqueueing work items...");
    try {
        producer_thread(filepath, work_queue, name_mapper, quant_resolver);
    } catch (const std::exception& e) {
        // Error in producer - shut down queue and wait for workers
        std::println(stderr, "  Producer error: {}", e.what());
        work_queue.shutdown();
        // No need to manually join workers; jthread destructor handles it
        throw;
    }

    // Producer finished - signal workers to finish
    work_queue.shutdown();
    std::println("  Producer finished, waiting for workers...");

    // ================================================================
    // Wait for All Workers
    // ================================================================
    // Although jthread joins on destruction, we need to wait explicitly here
    // to ensure results are ready before assembling them.
    for (auto& worker : workers) {
        worker.join();
    }

    std::println("  All workers finished, assembling results...");

    // ================================================================
    // Assemble Final Results
    // ================================================================
    WeightMap weights;
    std::unordered_map<std::string, QuantizedTensor> quantized_weights;

    size_t quantized_count = 0;
    for (auto& thread_results : all_results) {
        for (auto& result : thread_results) {
            weights[result.name] = std::move(result.tensor);

            if (result.quantized.has_value()) {
                quantized_weights[result.name] = std::move(*result.quantized);
                quantized_count++;
            }
        }
    }

    // Clear mmap handle now that all workers are done
    current_mmap_.reset();

    std::println("  ✓ Loaded {} tensors ({} quantized) using parallel pipeline",
                 weights.size(), quantized_count);

    return {std::move(weights), std::move(quantized_weights)};
}

void ParallelTensorLoader::producer_thread(
    const std::string& filepath,
    BoundedQueue<TensorWorkItem>& work_queue,
    std::function<std::string(const std::string&)> name_mapper,
    std::function<std::optional<quant::QuantType>(const std::string&)> quant_resolver
) {
    // ================================================================
    // Open and mmap file
    // ================================================================
    auto mmap = std::make_shared<MMapHandle>();
    if (!mmap->open(filepath)) {
        throw std::runtime_error("Failed to open/mmap file: " + filepath);
    }

    // Store mmap handle in member variable to keep it alive
    current_mmap_ = mmap;

    std::println("    Memory-mapped {:.2f} GB", mmap->size() / (1024.0 * 1024.0 * 1024.0));

    // ================================================================
    // Parse safetensors format using mmap
    // ================================================================
    // Use mmap_from_memory to parse from our already mmap-ed region
    // This avoids reading the file again and uses zero-copy access
    safetensors::safetensors_t st;
    std::string warn, err;

    bool ret = safetensors::mmap_from_memory(mmap->data(), mmap->size(), filepath, &st, &warn, &err);
    if (!ret) {
        throw std::runtime_error("Failed to parse safetensors: " + err);
    }

    if (!warn.empty()) {
        std::println("    Warning: {}", warn);
    }

    // Validate
    if (!safetensors::validate_data_offsets(st, err)) {
        throw std::runtime_error("Invalid data offsets: " + err);
    }

    std::println("    Found {} tensors in metadata", st.tensors.size());

    // ================================================================
    // Enqueue Work Items (with pointers to mmap region)
    // ================================================================
    size_t enqueued_count = 0;
    const auto& tensor_keys = st.tensors.keys();

    for (size_t i = 0; i < tensor_keys.size(); ++i) {
        const std::string& hf_name = tensor_keys[i];
        safetensors::tensor_t tensor_info;

        if (!st.tensors.at(hf_name, &tensor_info)) {
            continue;
        }

        // Map to internal name
        std::string internal_name = name_mapper(hf_name);
        if (internal_name.empty()) {
            // Skip unmapped tensors
            continue;
        }

        // Determine quantization type (if resolver provided)
        std::optional<quant::QuantType> quant_type = std::nullopt;
        if (quant_resolver) {
            quant_type = quant_resolver(internal_name);
        }

        // Get pointer to raw data in mmap region (zero-copy)
        // st.databuffer_addr points to the start of tensor data (after header)
        const uint8_t* raw_data = st.databuffer_addr + tensor_info.data_offsets[0];
        size_t data_size = tensor_info.data_offsets[1] - tensor_info.data_offsets[0];

        // Create work item with pointer to mmap region (not a copy)
        TensorWorkItem item(
            std::move(internal_name),
            tensor_info.shape,
            raw_data,  // Pointer to mmap region
            data_size,  // Size in bytes
            tensor_info.dtype,
            mmap,       // Shared pointer keeps mmap alive
            quant_type
        );

        // Push to queue (blocks if full - backpressure)
        if (!work_queue.push(std::move(item))) {
            // Queue was shut down (shouldn't happen in normal flow)
            break;
        }

        enqueued_count++;

        // Progress indicator
        if (enqueued_count % 50 == 0) {
            std::println("    Enqueued {} work items...", enqueued_count);
        }
    }

    std::println("    Producer finished: enqueued {} work items", enqueued_count);

    // Note: mmap handle is kept alive via shared_ptr in work items and current_mmap_
    // It will be released when all work items are processed and current_mmap_ is cleared
}

void ParallelTensorLoader::consumer_thread(
    BoundedQueue<TensorWorkItem>& work_queue,
    std::vector<TensorResult>& thread_results
) {
    try {
        while (true) {
            // Pop work item from queue
            auto maybe_item = work_queue.pop();
            if (!maybe_item) {
                // Queue is shutdown and empty - we're done
                break;
            }

            // Process the work item
            TensorResult result = process_work_item(*maybe_item);

            // Store result (local vector, no mutex needed)
            thread_results.push_back(std::move(result));
        }
    } catch (const std::exception& e) {
        // In a real app we should propagate this error to main thread
        // For now, just print and exit the thread
        std::println(stderr, "Worker thread error: {}", e.what());
        // Logic to stop other threads could be added here (e.g. shutdown queue)
        work_queue.shutdown();
    }
}

TensorResult ParallelTensorLoader::process_work_item(const TensorWorkItem& item) {
    // ================================================================
    // Create output tensor
    // ================================================================
    Tensor tensor(item.shape);
    size_t num_elements = tensor.size();

    // ================================================================
    // Convert data based on dtype (reading from mmap region safely)
    // ================================================================
    switch (item.dtype) {
        case safetensors::dtype::kFLOAT32: {
            // Direct copy for F32 - memcpy handles alignment automatically
            std::memcpy(tensor.data(), item.raw_data_ptr, num_elements * sizeof(float));
            break;
        }

        case safetensors::dtype::kBFLOAT16: {
            // Convert BF16 to F32
            const uint8_t* src_ptr = item.raw_data_ptr;
            float* f32_data = tensor.data();
            for (size_t i = 0; i < num_elements; ++i) {
                // Safe unaligned load
                uint16_t bf16_val = read_unaligned<uint16_t>(src_ptr + i * 2);
                f32_data[i] = bf16_to_f32(bf16_val);
            }
            break;
        }

        case safetensors::dtype::kFLOAT16: {
            // Convert FP16 to F32
            const uint8_t* src_ptr = item.raw_data_ptr;
            float* f32_data = tensor.data();
            for (size_t i = 0; i < num_elements; ++i) {
                // Safe unaligned load
                uint16_t fp16_val = read_unaligned<uint16_t>(src_ptr + i * 2);
                f32_data[i] = fp16_to_f32(fp16_val);
            }
            break;
        }

        default: {
            throw std::runtime_error(
                "Unsupported dtype for tensor: " + item.name
            );
        }
    }

    // ================================================================
    // Optionally quantize
    // ================================================================
    std::optional<QuantizedTensor> quantized = std::nullopt;
    if (item.quantize_to.has_value()) {
        // Quantize the F32 tensor
        quantized = QuantizedTensor::from_tensor(tensor, *item.quantize_to);
    }

    return TensorResult(item.name, std::move(tensor), std::move(quantized));
}

} // namespace freellm

