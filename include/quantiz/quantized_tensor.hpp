#pragma once

#include "../tensor.hpp"
#include "types.hpp"
#include "quantize.hpp"
#include "ops.hpp"
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace freellm {

// QuantizedTensor: Wrapper around quantized weight data
// This class stores tensors in quantized format (Q8_0 or Q4_K) and provides
// methods to dequantize them back to FP32 when needed.
//
// Design principles:
// - Non-invasive: doesn't modify existing Tensor class
// - Memory efficient: stores only quantized blocks
// - Easy to use: similar API to Tensor class
// - Type-safe: encapsulates quantization details
class QuantizedTensor {
public:
    // Constructors
    QuantizedTensor() = default;

    // Create quantized tensor from shape and type (allocates uninitialized memory)
    QuantizedTensor(const std::vector<size_t>& shape, quant::QuantType qtype)
        : shape_(shape), qtype_(qtype) {

        if (qtype == quant::QuantType::NONE) {
            throw std::invalid_argument("Use regular Tensor for non-quantized data");
        }

        // Calculate total elements
        num_elements_ = 1;
        for (size_t dim : shape) {
            num_elements_ *= dim;
        }

        const size_t row_size = row_size_from_shape(shape_);
        const size_t num_rows = num_rows_from_shape(shape_);
        const size_t row_bytes = quant::quant_row_bytes(qtype_, row_size);
        size_t memory_size = num_rows * row_bytes;

        data_ = std::make_unique<uint8_t[]>(memory_size);
        data_size_ = memory_size;
    }

    // Disable copy constructor and assignment (use .copy() method explicitly)
    QuantizedTensor(const QuantizedTensor&) = delete;
    QuantizedTensor& operator=(const QuantizedTensor&) = delete;

    // Enable move constructor and assignment
    QuantizedTensor(QuantizedTensor&& other) noexcept = default;
    QuantizedTensor& operator=(QuantizedTensor&& other) noexcept = default;

    // Create explicit copy
    QuantizedTensor copy() const {
        QuantizedTensor result(shape_, qtype_);
        std::memcpy(result.data_.get(), data_.get(), data_size_);
        return result;
    }

    // Factory: Create from existing FP32 tensor by quantizing it
    static QuantizedTensor from_tensor(const Tensor& tensor, quant::QuantType qtype) {
        if (tensor.dtype() != DType::Float32) {
            throw std::invalid_argument("Can only quantize Float32 tensors");
        }

        QuantizedTensor qtensor(tensor.shape(), qtype);

        // Quantize the data
        // For multi-dimensional tensors, quantize row by row
        const size_t num_rows = num_rows_from_shape(tensor.shape());
        const size_t row_size = row_size_from_shape(tensor.shape());

        const float* src = tensor.data();
        void* dst = qtensor.data();

        switch (qtype) {
            case quant::QuantType::Q8_0:
                if (num_rows == 0) {
                    break;
                }
                {
                    const size_t row_stride_bytes = quant::quant_row_bytes(qtype, row_size);
                    for (size_t row = 0; row < num_rows; ++row) {
                        quant::quantize_row_q8_0(
                            src + row * row_size,
                            static_cast<uint8_t*>(dst) + row * row_stride_bytes,
                            row_size
                        );
                    }
                }
                break;

            case quant::QuantType::Q4_K:
                if (num_rows == 0) {
                    break;
                }
                {
                    const size_t row_stride_bytes = quant::quant_row_bytes(qtype, row_size);
                    for (size_t row = 0; row < num_rows; ++row) {
                        quant::quantize_row_q4_K(
                            src + row * row_size,
                            static_cast<uint8_t*>(dst) + row * row_stride_bytes,
                            row_size
                        );
                    }
                }
                break;

            default:
                throw std::runtime_error("Unsupported quantization type");
        }

        return qtensor;
    }

    // Dequantize to FP32 tensor
    Tensor dequantize() const {
        if (qtype_ == quant::QuantType::NONE) {
            throw std::runtime_error("Cannot dequantize non-quantized tensor");
        }

        // Create output tensor with same shape
        Tensor result(shape_, DType::Float32);

        // Dequantize row by row
        const size_t num_rows = num_rows_from_shape(shape_);
        const size_t row_size = row_size_from_shape(shape_);

        const void* src = data_.get();
        float* dst = result.data();

        switch (qtype_) {
            case quant::QuantType::Q8_0:
                if (num_rows != 0) {
                    const size_t row_stride_bytes = quant::quant_row_bytes(qtype_, row_size);
                    for (size_t row = 0; row < num_rows; ++row) {
                        quant::dequantize_row_q8_0(
                            static_cast<const uint8_t*>(src) + row * row_stride_bytes,
                            dst + row * row_size,
                            row_size
                        );
                    }
                }
                break;

            case quant::QuantType::Q4_K:
                if (num_rows != 0) {
                    const size_t row_stride_bytes = quant::quant_row_bytes(qtype_, row_size);
                    for (size_t row = 0; row < num_rows; ++row) {
                        quant::dequantize_row_q4_K(
                            static_cast<const uint8_t*>(src) + row * row_stride_bytes,
                            dst + row * row_size,
                            row_size
                        );
                    }
                }
                break;

            default:
                throw std::runtime_error("Unsupported quantization type");
        }

        return result;
    }

    // Accessors
    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return num_elements_; }
    quant::QuantType qtype() const { return qtype_; }

    // Get raw pointer to quantized data
    void* data() { return data_.get(); }
    const void* data() const { return data_.get(); }

    // Get size of quantized data in bytes
    size_t data_size() const { return data_size_; }

    // Get number of blocks
    size_t num_blocks() const {
        const size_t row_size = row_size_from_shape(shape_);
        const size_t num_rows = num_rows_from_shape(shape_);
        return num_rows * quant::quant_blocks_per_row(qtype_, row_size);
    }

    // Type checking
    bool is_quantized() const {
        return qtype_ != quant::QuantType::NONE;
    }

    // Get block size for this quantization type
    size_t block_size() const {
        return quant::quant_block_size(qtype_);
    }

    // Get typed pointers to block data (for low-level operations)
    template<typename BlockType>
    BlockType* blocks() {
        return static_cast<BlockType*>(data());
    }

    template<typename BlockType>
    const BlockType* blocks() const {
        return static_cast<const BlockType*>(data());
    }

    // Specific block type accessors
    quant::block_q8_0* blocks_q8_0() {
        if (qtype_ != quant::QuantType::Q8_0) {
            throw std::runtime_error("Not a Q8_0 tensor");
        }
        return blocks<quant::block_q8_0>();
    }

    const quant::block_q8_0* blocks_q8_0() const {
        if (qtype_ != quant::QuantType::Q8_0) {
            throw std::runtime_error("Not a Q8_0 tensor");
        }
        return blocks<quant::block_q8_0>();
    }

    quant::block_q4_K* blocks_q4_K() {
        if (qtype_ != quant::QuantType::Q4_K) {
            throw std::runtime_error("Not a Q4_K tensor");
        }
        return blocks<quant::block_q4_K>();
    }

    const quant::block_q4_K* blocks_q4_K() const {
        if (qtype_ != quant::QuantType::Q4_K) {
            throw std::runtime_error("Not a Q4_K tensor");
        }
        return blocks<quant::block_q4_K>();
    }

    // Get memory usage info
    size_t memory_bytes() const {
        return data_size_;
    }

    float memory_mb() const {
        return static_cast<float>(data_size_) / (1024.0f * 1024.0f);
    }

    // Compare memory usage with FP32
    float compression_ratio() const {
        size_t fp32_size = num_elements_ * sizeof(float);
        return static_cast<float>(fp32_size) / static_cast<float>(data_size_);
    }

    // String representation
    std::string info() const {
        std::ostringstream oss;
        oss << "QuantizedTensor(";
        oss << "shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            oss << shape_[i];
            if (i < shape_.size() - 1) oss << ", ";
        }
        oss << "], ";
        oss << "dtype=" << quant::quant_type_name(qtype_) << ", ";
        oss << "size=" << num_elements_ << ", ";
        oss << "blocks=" << num_blocks() << ", ";
        oss << "memory=" << memory_mb() << " MB, ";
        oss << "compression=" << compression_ratio() << "x";
        oss << ")";
        return oss.str();
    }

private:
    std::vector<size_t> shape_;              // Tensor shape
    quant::QuantType qtype_ = quant::QuantType::NONE;  // Quantization type
    size_t num_elements_ = 0;                // Total number of elements
    size_t data_size_ = 0;                   // Size of quantized data in bytes
    std::unique_ptr<uint8_t[]> data_;        // Quantized block data

    static size_t num_rows_from_shape(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            return 1;
        }
        size_t rows = 1;
        for (size_t i = 0; i + 1 < shape.size(); ++i) {
            rows *= shape[i];
        }
        return rows;
    }

    static size_t row_size_from_shape(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            return 1;
        }
        return shape.back();
    }
};

} // namespace freellm

