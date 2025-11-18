#pragma once

#include <cstdint>
#include <string>

namespace freellm {
namespace cpu {

/**
 * @brief CPU feature flags detected at runtime
 */
struct CPUFeatures {
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512_vnni = false;
    bool neon = false;  // ARM NEON

    std::string cpu_brand;
};

/**
 * @brief Detect CPU features at runtime
 *
 * Uses CPUID instruction on x86/x64 or equivalent on ARM
 * This function should be called once at program startup
 */
CPUFeatures detect_cpu_features();

/**
 * @brief Get cached CPU features (singleton pattern)
 *
 * First call will detect features, subsequent calls return cached result
 */
const CPUFeatures& get_cpu_features();

/**
 * @brief Print detected CPU features to stdout
 */
void print_cpu_features(const CPUFeatures& features);

// ============================================================================
// Implementation
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

// x86/x64 CPUID implementation
namespace detail {

inline void cpuid(uint32_t eax, uint32_t ecx, uint32_t* regs) {
#if defined(_MSC_VER)
    __cpuidex(reinterpret_cast<int*>(regs), eax, ecx);
#elif defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__(
        "cpuid"
        : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
        : "a"(eax), "c"(ecx)
    );
#endif
}

inline void get_cpu_brand(char* brand) {
    uint32_t regs[4];
    for (uint32_t i = 0; i < 3; ++i) {
        cpuid(0x80000002 + i, 0, regs);
        for (int j = 0; j < 4; ++j) {
            *reinterpret_cast<uint32_t*>(brand + i * 16 + j * 4) = regs[j];
        }
    }
    brand[48] = '\0';
}

} // namespace detail

inline CPUFeatures detect_cpu_features() {
    CPUFeatures features;
    uint32_t regs[4];

    // Get CPU brand string
    char brand[49] = {0};
    detail::get_cpu_brand(brand);
    features.cpu_brand = std::string(brand);

    // Check basic features (leaf 1)
    detail::cpuid(1, 0, regs);
    features.avx = (regs[2] & (1 << 28)) != 0;  // ECX bit 28
    features.fma = (regs[2] & (1 << 12)) != 0;  // ECX bit 12

    // Check extended features (leaf 7)
    detail::cpuid(7, 0, regs);
    features.avx2 = (regs[1] & (1 << 5)) != 0;        // EBX bit 5
    features.avx512f = (regs[1] & (1 << 16)) != 0;    // EBX bit 16
    features.avx512_vnni = (regs[2] & (1 << 11)) != 0; // ECX bit 11

    return features;
}

#elif defined(__aarch64__) || defined(_M_ARM64)

// ARM implementation - most ARM64 CPUs have NEON
inline CPUFeatures detect_cpu_features() {
    CPUFeatures features;

    // On ARM64, NEON is mandatory
    features.neon = true;
    features.cpu_brand = "ARM64 CPU";

    // TODO: Could check for SVE, SVE2 on supported platforms
    // For now, we just assume NEON is available

    return features;
}

#else

// Fallback for unknown architectures
inline CPUFeatures detect_cpu_features() {
    CPUFeatures features;
    features.cpu_brand = "Unknown CPU";
    return features;
}

#endif

// Singleton accessor
inline const CPUFeatures& get_cpu_features() {
    static CPUFeatures features = detect_cpu_features();
    return features;
}

inline void print_cpu_features(const CPUFeatures& features) {
    #include <print>

    std::println("CPU Features Detected:");
    std::println("  Brand: {}", features.cpu_brand);
    std::println("  AVX:          {}", features.avx ? "Yes" : "No");
    std::println("  AVX2:         {}", features.avx2 ? "Yes" : "No");
    std::println("  FMA:          {}", features.fma ? "Yes" : "No");
    std::println("  AVX-512F:     {}", features.avx512f ? "Yes" : "No");
    std::println("  AVX-512 VNNI: {}", features.avx512_vnni ? "Yes" : "No");
    std::println("  ARM NEON:     {}", features.neon ? "Yes" : "No");
}

} // namespace cpu
} // namespace freellm
