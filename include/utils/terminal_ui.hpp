#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace freellm {
namespace ui {

// ========================================================================
// ANSI Color Codes for Terminal Styling
// ========================================================================

namespace colors {
    // Reset
    constexpr const char* RESET = "\033[0m";
    
    // Regular colors
    constexpr const char* BLACK = "\033[30m";
    constexpr const char* RED = "\033[31m";
    constexpr const char* GREEN = "\033[32m";
    constexpr const char* YELLOW = "\033[33m";
    constexpr const char* BLUE = "\033[34m";
    constexpr const char* MAGENTA = "\033[35m";
    constexpr const char* CYAN = "\033[36m";
    constexpr const char* WHITE = "\033[37m";
    
    // Bright/Bold colors
    constexpr const char* BRIGHT_BLACK = "\033[90m";
    constexpr const char* BRIGHT_RED = "\033[91m";
    constexpr const char* BRIGHT_GREEN = "\033[92m";
    constexpr const char* BRIGHT_YELLOW = "\033[93m";
    constexpr const char* BRIGHT_BLUE = "\033[94m";
    constexpr const char* BRIGHT_MAGENTA = "\033[95m";
    constexpr const char* BRIGHT_CYAN = "\033[96m";
    constexpr const char* BRIGHT_WHITE = "\033[97m";
    
    // Text styles
    constexpr const char* BOLD = "\033[1m";
    constexpr const char* DIM = "\033[2m";
    constexpr const char* ITALIC = "\033[3m";
    constexpr const char* UNDERLINE = "\033[4m";
    constexpr const char* BLINK = "\033[5m";
    
    // Background colors
    constexpr const char* BG_BLACK = "\033[40m";
    constexpr const char* BG_RED = "\033[41m";
    constexpr const char* BG_GREEN = "\033[42m";
    constexpr const char* BG_YELLOW = "\033[43m";
    constexpr const char* BG_BLUE = "\033[44m";
    constexpr const char* BG_MAGENTA = "\033[45m";
    constexpr const char* BG_CYAN = "\033[46m";
    constexpr const char* BG_WHITE = "\033[47m";
}

// ========================================================================
// ASCII Art Banners
// ========================================================================

inline void print_banner() {
    std::cout << "\n";
    std::cout << colors::BRIGHT_MAGENTA << "  ___            _    _    __  __ " << colors::RESET << "\n";
    std::cout << colors::BRIGHT_MAGENTA << " | __| _ ___ ___| |  | |  |  \\/  |" << colors::RESET << "\n";
    std::cout << colors::BRIGHT_MAGENTA << " | _| '_/ -_) -_) |__| |__| |\\/| |" << colors::RESET << "\n";
    std::cout << colors::BRIGHT_MAGENTA << " |_||_| \\___\\___|____|____|_|  |_|" << colors::RESET << "\n";
    std::cout << colors::DIM << "  Local LLM • Metal Accelerated • Quantized" << colors::RESET << "\n";
    std::cout << "\n";
}

inline void print_section_header(const std::string& title) {
    std::cout << "\n" << colors::DIM << "  ──────────────────────────────────────────" << colors::RESET << "\n";
    std::cout << colors::BRIGHT_CYAN << "  ◆ " << colors::BRIGHT_WHITE << colors::BOLD << title << colors::RESET << "\n";
}

inline void print_subsection(const std::string& title) {
    std::cout << colors::BRIGHT_BLUE << "  ┌─ " << colors::BRIGHT_WHITE << title << colors::RESET << "\n";
}

// ========================================================================
// Status Indicators
// ========================================================================

inline void print_status(const std::string& message) {
    std::cout << colors::DIM << "  │ " << colors::RESET << message << "\n";
}

inline void print_success(const std::string& message) {
    std::cout << colors::BRIGHT_GREEN << "  ✓ " << colors::RESET << message << "\n";
}

inline void print_warning(const std::string& message) {
    std::cout << colors::BRIGHT_YELLOW << "  ⚠ " << colors::RESET << message << "\n";
}

inline void print_error(const std::string& message) {
    std::cout << colors::BRIGHT_RED << "  ✗ " << colors::RESET << message << "\n";
}

inline void print_info(const std::string& message) {
    std::cout << colors::BRIGHT_CYAN << "  ◆ " << colors::RESET << message << "\n";
}

inline void print_loading(const std::string& message) {
    std::cout << colors::BRIGHT_MAGENTA << "  ⟳ " << colors::RESET << message << "\n";
}

// ========================================================================
// Progress Indicators
// ========================================================================

inline void print_progress_bar(size_t current, size_t total, size_t width = 30) {
    float progress = static_cast<float>(current) / static_cast<float>(total);
    size_t filled = static_cast<size_t>(progress * width);
    
    std::cout << colors::DIM << "  │ " << colors::RESET;
    std::cout << colors::BRIGHT_CYAN << "[" << colors::RESET;
    
    for (size_t i = 0; i < width; ++i) {
        if (i < filled) {
            std::cout << colors::BRIGHT_MAGENTA << "█" << colors::RESET;
        } else if (i == filled) {
            std::cout << colors::BRIGHT_CYAN << "▓" << colors::RESET;
        } else {
            std::cout << colors::DIM << "░" << colors::RESET;
        }
    }
    
    std::cout << colors::BRIGHT_CYAN << "]" << colors::RESET;
    std::cout << colors::BRIGHT_WHITE << " " << std::fixed << std::setprecision(1) 
              << (progress * 100.0f) << "%" << colors::RESET;
    std::cout << " (" << current << "/" << total << ")\r" << std::flush;
}

inline void print_progress_complete(size_t total) {
    std::cout << "\033[2K"; // Clear line
    print_success(std::to_string(total) + " items processed");
}

// ========================================================================
// Model Information Display
// ========================================================================

inline void print_model_config(const std::string& model_name, 
                                size_t d_model, size_t n_layers, 
                                size_t n_heads, size_t vocab_size,
                                size_t max_seq_len) {
    std::cout << colors::BRIGHT_BLUE << "  ┌─ " << colors::BRIGHT_WHITE << "Model Architecture" << colors::RESET << "\n";
    std::cout << colors::DIM << "  │ " << colors::RESET << colors::BRIGHT_CYAN     << "Name:       " << colors::RESET << model_name << "\n";
    std::cout << colors::DIM << "  │ " << colors::RESET << colors::BRIGHT_CYAN     << "Dim:        " << colors::RESET << d_model << "\n";
    std::cout << colors::DIM << "  │ " << colors::RESET << colors::BRIGHT_CYAN     << "Layers:     " << colors::RESET << n_layers << "\n";
    std::cout << colors::DIM << "  │ " << colors::RESET << colors::BRIGHT_CYAN     << "Heads:      " << colors::RESET << n_heads << "\n";
    std::cout << colors::DIM << "  │ " << colors::RESET << colors::BRIGHT_CYAN     << "Vocab:      " << colors::RESET << vocab_size << "\n";
    std::cout << colors::DIM << "  │ " << colors::RESET << colors::BRIGHT_CYAN     << "Max Seq:    " << colors::RESET << max_seq_len << "\n";
    std::cout << colors::DIM << "  └─────────────────────────────" << colors::RESET << "\n";
}

// ========================================================================
// File Loading Display
// ========================================================================

inline void print_file_info(const std::string& filename, double size_gb, size_t tensor_count) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << size_gb << " GB";
    std::cout << colors::DIM << "  │ " << colors::RESET 
              << colors::BRIGHT_WHITE << filename << colors::RESET
              << colors::DIM << " (" << ss.str() << ", " << tensor_count << " tensors)" << colors::RESET << "\n";
}

// ========================================================================
// Metal/GPU Status
// ========================================================================

inline void print_metal_status(const std::string& operation, const std::string& details = "") {
    std::cout << colors::BRIGHT_BLUE << "  ⚡ " << colors::RESET 
              << colors::BRIGHT_YELLOW << "[Metal] " << colors::RESET 
              << operation;
    if (!details.empty()) {
        std::cout << colors::DIM << " → " << details << colors::RESET;
    }
    std::cout << "\n";
}

// ========================================================================
// Inference Output
// ========================================================================

inline void print_inference_header() {
    std::cout << "\n" << colors::DIM << "  ──────────────────────────────────────────" << colors::RESET << "\n";
    std::cout << colors::BRIGHT_CYAN << "  ◆ " << colors::BRIGHT_WHITE << colors::BOLD << "INFERENCE OUTPUT" << colors::RESET << "\n";
}

inline void print_sequence_output(size_t seq_id, const std::string& text) {
    std::cout << colors::BRIGHT_MAGENTA << "  [" << seq_id << "] " << colors::RESET
              << colors::BRIGHT_WHITE << text << colors::RESET;
}

inline void print_inference_footer() {
    std::cout << "\n" << colors::DIM << "  ──────────────────────────────────────────" << colors::RESET << "\n";
}

inline void print_inference_complete(double tokens_per_sec = 0.0) {
    std::cout << "\n" << colors::BRIGHT_GREEN << "  ✓ " << colors::RESET 
              << colors::BRIGHT_WHITE << "Inference Complete" << colors::RESET;
    if (tokens_per_sec > 0) {
        std::cout << colors::DIM << " (" << std::fixed << std::setprecision(1) 
                  << tokens_per_sec << " tok/s)" << colors::RESET;
    }
    std::cout << "\n";
}

// ========================================================================
// Chat Mode Display
// ========================================================================

inline void print_chat_prompt() {
    std::cout << "\n" << colors::BRIGHT_CYAN << "╭─" << colors::BRIGHT_WHITE << " You " << colors::BRIGHT_CYAN << "───────────────────────────────────────────────────────╮" << colors::RESET << "\n";
    std::cout << colors::BRIGHT_CYAN << "│ " << colors::RESET;
}

inline void print_assistant_header() {
    std::cout << "\n" << colors::BRIGHT_MAGENTA << "╭─" << colors::BRIGHT_WHITE << " Assistant " << colors::BRIGHT_MAGENTA << "────────────────────────────────────────────────╮" << colors::RESET << "\n";
    std::cout << colors::BRIGHT_MAGENTA << "│ " << colors::RESET;
}

inline void print_thinking() {
    std::cout << colors::DIM << colors::ITALIC << "  ◌ Thinking..." << colors::RESET << std::flush;
}

inline void clear_thinking() {
    std::cout << "\r\033[K"; // Clear line
}

} // namespace ui
} // namespace freellm
