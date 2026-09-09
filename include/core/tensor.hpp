#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdlib>

namespace freellm {

/**
 * @brief Data type enumeration for tensor elements
 */
enum class DType {
    Float32,
    Float16,  // Future support
    Int8,     // Future quantization support
    Int4      // Future quantization support
};

/**
 * @brief Get size in bytes for each data type
 */
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int8:    return 1;
        case DType::Int4:    return 1; // Packed, but minimal unit is 1 byte
        default: return 4;
    }
}

/**
 * @brief Internal helpers for raw row-major access
 *
 * These are zero-cost inline utilities intended for use in hot paths.
 */
namespace detail {
    template <typename T>
    inline const T* row_ptr(const T* data, size_t rowIndex, size_t rowStride) {
        return data + rowIndex * rowStride;
    }

    template <typename T>
    inline T* row_ptr(T* data, size_t rowIndex, size_t rowStride) {
        return data + rowIndex * rowStride;
    }
}

/**
 * @brief A lightweight tensor class for LLM inference
 *
 * Design principles:
 * - Contiguous memory layout for cache efficiency
 * - SIMD-aligned allocation (64-byte for AVX-512)
 * - Move semantics for efficient memory management
 * - Row-major order (C-style) for better CPU cache behavior
 *
 * Memory layout:
 * - For shape [2, 3, 4], strides are [12, 4, 1]
 * - Element at indices [i, j, k] is at: data[i*12 + j*4 + k]
 */
class Tensor {
public:
    /**
     * @brief Default constructor - creates empty tensor
     */
    Tensor()
        : shape_{}, strides_{}, size_(0), data_(nullptr, &std::free), dtype_(DType::Float32) {}

    /**
     * @brief Construct tensor with given shape
     * @param shape Dimensions of the tensor
     * @param dtype Data type of tensor elements (default: Float32)
     */
    explicit Tensor(std::vector<size_t> shape, DType dtype = DType::Float32)
        : shape_(std::move(shape)), dtype_(dtype) {

        // Compute total size
        size_ = std::accumulate(shape_.begin(), shape_.end(),
                               size_t(1), std::multiplies<>());

        // Compute strides (row-major order)
        compute_strides();

        // Allocate aligned memory for SIMD operations
        allocate_memory();
    }

    /**
     * @brief Construct tensor with shape and initialize with value
     */
    Tensor(std::vector<size_t> shape, float init_value, DType dtype = DType::Float32)
        : Tensor(std::move(shape), dtype) {
        fill(init_value);
    }

    /**
     * @brief Construct tensor from existing data (copies data)
     */
    Tensor(std::vector<size_t> shape, const float* data, DType dtype = DType::Float32)
        : Tensor(std::move(shape), dtype) {
        std::memcpy(data_.get(), data, size_ * sizeof(float));
    }

    // Disable copy constructor and assignment (use explicit copy method)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Enable move semantics
    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_))
        , strides_(std::move(other.strides_))
        , size_(other.size_)
        , data_(std::move(other.data_))
        , dtype_(other.dtype_) {
        other.size_ = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            size_ = other.size_;
            data_ = std::move(other.data_);
            dtype_ = other.dtype_;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Create a deep copy of this tensor
     */
    Tensor copy() const {
        Tensor result(shape_, dtype_);
        std::memcpy(result.data_.get(), data_.get(), size_ * dtype_size(dtype_));
        return result;
    }

    /**
     * @brief Alias for copy() - creates a deep copy of this tensor
     */
    Tensor clone() const { return copy(); }

    // Accessors
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    DType dtype() const { return dtype_; }

    float* data() { return reinterpret_cast<float*>(data_.get()); }
    const float* data() const { return reinterpret_cast<const float*>(data_.get()); }

    /**
     * @brief Access element at flat index (no bounds checking for performance)
     */
    float& operator[](size_t idx) { return data()[idx]; }
    const float& operator[](size_t idx) const { return data()[idx]; }

    /**
     * @brief Access element with multi-dimensional indices
     * Example: tensor.at({0, 1, 2}) for 3D tensor
     */
    float& at(const std::vector<size_t>& indices) {
        return data()[compute_offset(indices)];
    }

    const float& at(const std::vector<size_t>& indices) const {
        return data()[compute_offset(indices)];
    }

    /**
     * @brief Fill tensor with a constant value
     */
    void fill(float value) {
        float* ptr = data();
        for (size_t i = 0; i < size_; ++i) {
            ptr[i] = value;
        }
    }

    /**
     * @brief Zero out all elements
     */
    void zero() {
        std::memset(data_.get(), 0, size_ * dtype_size(dtype_));
    }

    /**
     * @brief Reshape tensor (must preserve total size)
     */
    void reshape(std::vector<size_t> new_shape) {
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                               size_t(1), std::multiplies<>());
        if (new_size != size_) {
            throw std::invalid_argument(
                "Reshape: new shape must have same total size. "
                "Current: " + std::to_string(size_) +
                ", New: " + std::to_string(new_size));
        }
        shape_ = std::move(new_shape);
        compute_strides();
    }

    /**
     * @brief Check if tensor is empty
     */
    bool empty() const { return size_ == 0; }

    /**
     * @brief Print tensor information (for debugging)
     */
    void print_info() const;

private:
    std::vector<size_t> shape_;     // Dimensions: [dim0, dim1, dim2, ...]
    std::vector<size_t> strides_;   // Strides for computing flat index
    size_t size_;                    // Total number of elements
    std::unique_ptr<uint8_t[], decltype(&std::free)> data_{nullptr, &std::free};
    DType dtype_;

    /**
     * @brief Compute strides for row-major ordering
     * For shape [d0, d1, d2], strides are [d1*d2, d2, 1]
     */
    void compute_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) return;

        strides_.back() = 1;
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    /**
     * @brief Allocate SIMD-aligned memory
     * 64-byte alignment for AVX-512 (8 floats * 8 bytes = 64)
     */
    void allocate_memory() {
        if (size_ == 0) return;

        constexpr size_t alignment = 64;
        size_t bytes = size_ * dtype_size(dtype_);

        // aligned_alloc requires size to be multiple of alignment
        size_t aligned_bytes = ((bytes + alignment - 1) / alignment) * alignment;

        void* ptr = std::aligned_alloc(alignment, aligned_bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }

        data_ = std::unique_ptr<uint8_t[], decltype(&std::free)>(
            static_cast<uint8_t*>(ptr), &std::free);

        // Initialize to zero for safety
        std::memset(data_.get(), 0, aligned_bytes);
    }

    /**
     * @brief Compute flat offset from multi-dimensional indices
     */
    size_t compute_offset(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Number of indices must match tensor dimensions");
        }

        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            offset += indices[i] * strides_[i];
        }
        return offset;
    }
};

} // namespace freellm
