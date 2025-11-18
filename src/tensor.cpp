// Define safetensors implementation once in this translation unit
#define SAFETENSORS_CPP_IMPLEMENTATION

// Disable warnings for third-party header
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#include "safetensors.hh"
#pragma GCC diagnostic pop

#include "tensor.hpp"
#include <iostream>
#include <iomanip>

namespace freellm {

void Tensor::print_info() const {
    std::cout << "Tensor Information:\n";
    std::cout << "  Shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "  Strides: [";
    for (size_t i = 0; i < strides_.size(); ++i) {
        std::cout << strides_[i];
        if (i < strides_.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "  Size: " << size_ << "\n";
    std::cout << "  Dimensions: " << ndim() << "\n";

    const char* dtype_str = "Unknown";
    switch (dtype_) {
        case DType::Float32: dtype_str = "Float32"; break;
        case DType::Float16: dtype_str = "Float16"; break;
        case DType::Int8:    dtype_str = "Int8"; break;
        case DType::Int4:    dtype_str = "Int4"; break;
    }
    std::cout << "  DType: " << dtype_str << "\n";

    // Print first few elements if tensor is small
    if (size_ > 0 && size_ <= 20) {
        std::cout << "  Data: [";
        const float* d = data();
        for (size_t i = 0; i < size_; ++i) {
            std::cout << std::fixed << std::setprecision(4) << d[i];
            if (i < size_ - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    } else if (size_ > 20) {
        std::cout << "  Data: [";
        const float* d = data();
        for (size_t i = 0; i < 5; ++i) {
            std::cout << std::fixed << std::setprecision(4) << d[i] << ", ";
        }
        std::cout << "... ";
        for (size_t i = size_ - 3; i < size_; ++i) {
            std::cout << std::fixed << std::setprecision(4) << d[i];
            if (i < size_ - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

} // namespace freellm
